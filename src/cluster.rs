use crate::geometry::{AttributeStream, PackedPrimitive, PreparedPrimitive};
use anyhow::{Result, bail};
use meshopt::{
    LocalizedMeshlets, Meshlets, VertexDataAdapter, build_meshlets, ffi, optimize_vertex_cache,
};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ClusterConfig {
    pub max_vertices: usize,
    pub max_triangles: usize,
    pub cone_weight: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 64,
            cone_weight: 0.0,
        }
    }
}

impl ClusterConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_vertices == 0 || self.max_vertices > 256 {
            bail!("max_vertices must be in 1..=256");
        }
        if self.max_triangles == 0
            || self.max_triangles > 512
            || !self.max_triangles.is_multiple_of(4)
        {
            bail!("max_triangles must be in 4..=512 and divisible by 4");
        }
        if !(0.0..=1.0).contains(&self.cone_weight) {
            bail!("cone_weight must be in 0..=1");
        }
        Ok(())
    }
}

pub struct LocalizedStreams {
    pub positions: LocalizedMeshlets,
    pub attributes: Vec<AttributeStream>,
}

impl LocalizedStreams {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

pub struct PrimitiveClusters {
    pub standard: Meshlets,
    pub localized: LocalizedStreams,
    pub referenced_vertex_count: usize,
    pub max_vertices: usize,
}

pub fn prepare_primitive(primitive: &PackedPrimitive) -> PreparedPrimitive {
    if primitive.indices.is_empty() {
        return PreparedPrimitive {
            name: primitive.name.clone(),
            vertex_count: 0,
            streams: Vec::new(),
            indices: Vec::new(),
        };
    }

    let cache_indices = optimize_vertex_cache(&primitive.indices, primitive.vertex_count);
    let (new_vertex_count, remap) = vertex_fetch_remap(&cache_indices, primitive.vertex_count);

    let streams = primitive
        .streams
        .iter()
        .map(|stream| remap_stream(stream, new_vertex_count, &remap))
        .collect();

    let indices = cache_indices
        .into_iter()
        .map(|index| remap[index as usize])
        .collect::<Vec<_>>();

    PreparedPrimitive {
        name: primitive.name.clone(),
        vertex_count: new_vertex_count,
        streams,
        indices,
    }
}

fn remap_stream(stream: &AttributeStream, vertex_count: usize, remap: &[u32]) -> AttributeStream {
    let element_size = stream.element_size();
    let mut data = vec![0u8; vertex_count * element_size];

    for (old_index, &new_index) in remap.iter().enumerate() {
        if new_index != u32::MAX {
            let source = old_index * element_size;
            let destination = new_index as usize * element_size;
            data[destination..destination + element_size]
                .copy_from_slice(&stream.data[source..source + element_size]);
        }
    }

    AttributeStream {
        name: stream.name.clone(),
        components: stream.components,
        data,
    }
}

fn vertex_fetch_remap(indices: &[u32], vertex_count: usize) -> (usize, Vec<u32>) {
    let mut remap = vec![u32::MAX; vertex_count];
    let new_vertex_count = unsafe {
        ffi::meshopt_optimizeVertexFetchRemap(
            remap.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        )
    };
    (new_vertex_count, remap)
}

pub fn build_clusters(
    primitive: &PreparedPrimitive,
    config: ClusterConfig,
) -> Result<PrimitiveClusters> {
    config.validate()?;

    let position = primitive.position();
    let position_size = position.element_size();
    let adapter = VertexDataAdapter::new(&position.data, position_size, 0)?;
    let standard = build_meshlets(
        &primitive.indices,
        &adapter,
        config.max_vertices,
        config.max_triangles,
        config.cone_weight,
    );
    let localized_positions = standard.localize(&adapter, position_size);
    let localized_attributes = localize_attributes(
        primitive,
        &standard,
        localized_positions.vertices.len() / position_size,
    );

    let mut referenced = HashSet::new();
    for &vertex in &standard.vertices {
        referenced.insert(vertex);
    }

    Ok(PrimitiveClusters {
        standard,
        localized: LocalizedStreams {
            positions: localized_positions,
            attributes: localized_attributes,
        },
        referenced_vertex_count: referenced.len(),
        max_vertices: config.max_vertices,
    })
}

fn localize_attributes(
    primitive: &PreparedPrimitive,
    standard: &Meshlets,
    localized_vertex_count: usize,
) -> Vec<AttributeStream> {
    primitive
        .streams
        .iter()
        .skip(1)
        .map(|stream| {
            let element_size = stream.element_size();
            let mut data = Vec::with_capacity(localized_vertex_count * element_size);

            for meshlet in &standard.meshlets {
                let vertex_offset = meshlet.vertex_offset as usize;
                let vertex_count = meshlet.vertex_count as usize;
                for &global_vertex in
                    &standard.vertices[vertex_offset..vertex_offset + vertex_count]
                {
                    let source = global_vertex as usize * element_size;
                    data.extend_from_slice(&stream.data[source..source + element_size]);
                }
            }

            debug_assert_eq!(data.len(), localized_vertex_count * element_size);

            AttributeStream {
                name: stream.name.clone(),
                components: stream.components,
                data,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(name: &str, components: usize, values: &[f32]) -> AttributeStream {
        let mut data = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            data.extend_from_slice(&value.to_ne_bytes());
        }
        AttributeStream {
            name: name.to_string(),
            components,
            data,
        }
    }

    #[test]
    fn prepare_and_localize_keep_streams_separate() {
        let primitive = PackedPrimitive {
            name: "quad".to_string(),
            vertex_count: 4,
            streams: vec![
                stream(
                    "POSITION",
                    3,
                    &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                ),
                stream(
                    "NORMAL",
                    3,
                    &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                ),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        };

        let prepared = prepare_primitive(&primitive);
        assert_eq!(prepared.streams.len(), 2);
        assert_eq!(prepared.streams[0].vertex_count(), prepared.vertex_count);
        assert_eq!(prepared.streams[1].vertex_count(), prepared.vertex_count);

        let clusters = build_clusters(
            &prepared,
            ClusterConfig {
                max_vertices: 4,
                max_triangles: 4,
                cone_weight: 0.0,
            },
        )
        .unwrap();

        assert_eq!(clusters.localized.attributes.len(), 1);
        assert_eq!(
            clusters.localized.attributes[0].vertex_count(),
            clusters.localized.positions.vertices.len() / 12
        );
    }
}

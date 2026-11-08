use crate::geometry::{PackedPrimitive, PreparedPrimitive};
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

pub struct PrimitiveClusters {
    pub standard: Meshlets,
    pub localized: LocalizedMeshlets,
    pub referenced_vertex_count: usize,
}

pub fn prepare_primitive(primitive: &PackedPrimitive) -> PreparedPrimitive {
    if primitive.indices.is_empty() {
        return PreparedPrimitive {
            name: primitive.name.clone(),
            vertices: Vec::new(),
            vertex_count: 0,
            stride: primitive.stride,
            attributes: primitive.attributes.clone(),
            indices: Vec::new(),
        };
    }

    let cache_indices = optimize_vertex_cache(&primitive.indices, primitive.vertex_count);
    let (new_vertex_count, remap) = vertex_fetch_remap(&cache_indices, primitive.vertex_count);

    let mut vertices = vec![0u8; new_vertex_count * primitive.stride];
    for (old_index, &new_index) in remap.iter().enumerate() {
        if new_index != u32::MAX {
            let source = old_index * primitive.stride;
            let destination = new_index as usize * primitive.stride;
            vertices[destination..destination + primitive.stride]
                .copy_from_slice(&primitive.vertices[source..source + primitive.stride]);
        }
    }

    let indices = cache_indices
        .into_iter()
        .map(|index| remap[index as usize])
        .collect::<Vec<_>>();

    PreparedPrimitive {
        name: primitive.name.clone(),
        vertices,
        vertex_count: new_vertex_count,
        stride: primitive.stride,
        attributes: primitive.attributes.clone(),
        indices,
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

    let adapter = VertexDataAdapter::new(&primitive.vertices, primitive.stride, 0)?;
    let standard = build_meshlets(
        &primitive.indices,
        &adapter,
        config.max_vertices,
        config.max_triangles,
        config.cone_weight,
    );
    let localized = standard.localize(&adapter, primitive.stride);

    let mut referenced = HashSet::new();
    for &vertex in &standard.vertices {
        referenced.insert(vertex);
    }

    Ok(PrimitiveClusters {
        standard,
        localized,
        referenced_vertex_count: referenced.len(),
    })
}

use crate::geometry::{AttributeStream, PackedPrimitive};
use anyhow::{Context, Result, anyhow, bail};
use gltf::mesh::Mode;
use std::mem::size_of;
use std::path::Path;

type Mat4 = [[f32; 4]; 4];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub struct LoadedGltf {
    pub primitives: Vec<PackedPrimitive>,
    pub warnings: Vec<String>,
}

pub fn load_gltf(path: &Path) -> Result<LoadedGltf> {
    let gltf::Gltf { document, blob } =
        gltf::Gltf::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let buffers = gltf::import_buffers(&document, path.parent(), blob)
        .with_context(|| format!("failed to import buffers from {}", path.display()))?;

    let mut primitives = Vec::new();
    let mut warnings = Vec::new();

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());

    if let Some(scene) = scene {
        for node in scene.nodes() {
            visit_node(node, IDENTITY, &buffers, &mut primitives, &mut warnings);
        }
    } else {
        for mesh in document.meshes() {
            let name = mesh.name().unwrap_or("mesh");
            for primitive in mesh.primitives() {
                collect_primitive(
                    &primitive,
                    IDENTITY,
                    &buffers,
                    name,
                    &mut primitives,
                    &mut warnings,
                );
            }
        }
    }

    Ok(LoadedGltf {
        primitives,
        warnings,
    })
}

fn visit_node(
    node: gltf::Node<'_>,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    primitives: &mut Vec<PackedPrimitive>,
    warnings: &mut Vec<String>,
) {
    let world = mat4_mul(&parent, &node.transform().matrix());

    if let Some(mesh) = node.mesh() {
        let name = mesh.name().unwrap_or("mesh");
        for primitive in mesh.primitives() {
            collect_primitive(&primitive, world, buffers, name, primitives, warnings);
        }
    }

    for child in node.children() {
        visit_node(child, world, buffers, primitives, warnings);
    }
}

fn collect_primitive(
    primitive: &gltf::Primitive<'_>,
    world: Mat4,
    buffers: &[gltf::buffer::Data],
    mesh_name: &str,
    primitives: &mut Vec<PackedPrimitive>,
    warnings: &mut Vec<String>,
) {
    match pack_primitive(primitive, &world, buffers, mesh_name) {
        Ok(primitive) => primitives.push(primitive),
        Err(error) => warnings.push(format!("skipping {mesh_name}: {error:#}")),
    }
}

fn pack_primitive(
    primitive: &gltf::Primitive<'_>,
    world: &Mat4,
    buffers: &[gltf::buffer::Data],
    mesh_name: &str,
) -> Result<PackedPrimitive> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions = reader
        .read_positions()
        .ok_or_else(|| anyhow!("primitive has no POSITION attribute"))?
        .collect::<Vec<_>>();
    if positions.is_empty() {
        bail!("primitive has an empty POSITION attribute");
    }

    let normals = reader.read_normals().map(|iter| iter.collect::<Vec<_>>());
    let tex_coords = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect::<Vec<_>>());
    let tangents = reader.read_tangents().map(|iter| iter.collect::<Vec<_>>());
    let colors = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect::<Vec<_>>());

    let raw_indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<_>>(),
        None => (0..positions.len() as u32).collect(),
    };
    let indices = triangle_list(primitive.mode(), &raw_indices)?;

    let vertex_count = positions.len();
    for &index in &indices {
        if index as usize >= vertex_count {
            bail!("index {index} is out of range for {vertex_count} vertices");
        }
    }

    if let Some(normals) = &normals {
        if normals.len() != vertex_count {
            bail!(
                "NORMAL count {} does not match POSITION count {vertex_count}",
                normals.len()
            );
        }
    }
    if let Some(tex_coords) = &tex_coords {
        if tex_coords.len() != vertex_count {
            bail!(
                "TEXCOORD_0 count {} does not match POSITION count {vertex_count}",
                tex_coords.len()
            );
        }
    }
    if let Some(tangents) = &tangents {
        if tangents.len() != vertex_count {
            bail!(
                "TANGENT count {} does not match POSITION count {vertex_count}",
                tangents.len()
            );
        }
    }
    if let Some(colors) = &colors {
        if colors.len() != vertex_count {
            bail!(
                "COLOR_0 count {} does not match POSITION count {vertex_count}",
                colors.len()
            );
        }
    }

    let transformed_positions = positions
        .iter()
        .map(|position| transform_position(world, *position))
        .collect::<Vec<_>>();
    let mut streams = vec![AttributeStream {
        name: "POSITION".to_string(),
        components: 3,
        data: pack_f32_stream(&transformed_positions),
    }];

    if let Some(normals) = &normals {
        let transformed = normals
            .iter()
            .map(|normal| transform_normal(world, *normal))
            .collect::<Vec<_>>();
        streams.push(AttributeStream {
            name: "NORMAL".to_string(),
            components: 3,
            data: pack_f32_stream(&transformed),
        });
    }

    if let Some(tex_coords) = &tex_coords {
        streams.push(AttributeStream {
            name: "TEXCOORD_0".to_string(),
            components: 2,
            data: pack_f32_stream(tex_coords),
        });
    }

    if let Some(tangents) = &tangents {
        let transformed = tangents
            .iter()
            .map(|tangent| transform_tangent(world, *tangent))
            .collect::<Vec<_>>();
        streams.push(AttributeStream {
            name: "TANGENT".to_string(),
            components: 4,
            data: pack_f32_stream(&transformed),
        });
    }

    if let Some(colors) = &colors {
        streams.push(AttributeStream {
            name: "COLOR_0".to_string(),
            components: 4,
            data: pack_f32_stream(colors),
        });
    }

    Ok(PackedPrimitive {
        name: format!("{mesh_name}[{}]", primitive.index()),
        vertex_count,
        streams,
        indices,
    })
}

fn pack_f32_stream<const N: usize>(values: &[[f32; N]]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * N * size_of::<f32>());
    for value in values {
        for component in value {
            data.extend_from_slice(&component.to_ne_bytes());
        }
    }
    data
}

fn triangle_list(mode: Mode, indices: &[u32]) -> Result<Vec<u32>> {
    match mode {
        Mode::Triangles => Ok(indices.to_vec()),
        Mode::TriangleStrip => {
            if indices.len() < 3 {
                return Ok(Vec::new());
            }
            let mut triangles = Vec::with_capacity((indices.len() - 2) * 3);
            for i in 0..indices.len() - 2 {
                if i % 2 == 0 {
                    triangles.extend_from_slice(&[indices[i], indices[i + 1], indices[i + 2]]);
                } else {
                    triangles.extend_from_slice(&[indices[i + 1], indices[i], indices[i + 2]]);
                }
            }
            Ok(triangles)
        }
        Mode::TriangleFan => {
            if indices.len() < 3 {
                return Ok(Vec::new());
            }
            let mut triangles = Vec::with_capacity((indices.len() - 2) * 3);
            for i in 1..indices.len() - 1 {
                triangles.extend_from_slice(&[indices[0], indices[i], indices[i + 1]]);
            }
            Ok(triangles)
        }
        other => bail!("unsupported primitive mode {other:?}"),
    }
}

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut result = [[0.0f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            let mut value = 0.0;
            for k in 0..4 {
                value += a[k][row] * b[column][k];
            }
            result[column][row] = value;
        }
    }
    result
}

fn transform_position(matrix: &Mat4, position: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * position[0]
            + matrix[1][0] * position[1]
            + matrix[2][0] * position[2]
            + matrix[3][0],
        matrix[0][1] * position[0]
            + matrix[1][1] * position[1]
            + matrix[2][1] * position[2]
            + matrix[3][1],
        matrix[0][2] * position[0]
            + matrix[1][2] * position[1]
            + matrix[2][2] * position[2]
            + matrix[3][2],
    ]
}

fn transform_normal(matrix: &Mat4, normal: [f32; 3]) -> [f32; 3] {
    let a = [matrix[0][0], matrix[0][1], matrix[0][2]];
    let b = [matrix[1][0], matrix[1][1], matrix[1][2]];
    let c = [matrix[2][0], matrix[2][1], matrix[2][2]];
    let determinant = dot(a, cross(b, c));

    let transformed = if determinant.abs() < 1.0e-12 {
        linear_transform(matrix, normal)
    } else {
        let inverse_determinant = 1.0 / determinant;
        add(
            scale(cross(b, c), normal[0] * inverse_determinant),
            add(
                scale(cross(c, a), normal[1] * inverse_determinant),
                scale(cross(a, b), normal[2] * inverse_determinant),
            ),
        )
    };

    normalize(transformed)
}

fn transform_tangent(matrix: &Mat4, tangent: [f32; 4]) -> [f32; 4] {
    let xyz = normalize(linear_transform(
        matrix,
        [tangent[0], tangent[1], tangent[2]],
    ));
    [xyz[0], xyz[1], xyz[2], tangent[3]]
}

fn linear_transform(matrix: &Mat4, value: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * value[0] + matrix[1][0] * value[1] + matrix[2][0] * value[2],
        matrix[0][1] * value[0] + matrix[1][1] * value[1] + matrix[2][1] * value[2],
        matrix[0][2] * value[0] + matrix[1][2] * value[1] + matrix[2][2] * value[2],
    ]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], factor: f32) -> [f32; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let length = dot(a, a).sqrt();
    if length > 1.0e-12 {
        scale(a, 1.0 / length)
    } else {
        a
    }
}

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct VertexAttribute {
    pub name: String,
    pub components: usize,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct PackedPrimitive {
    pub name: String,
    pub vertices: Vec<u8>,
    pub vertex_count: usize,
    pub stride: usize,
    pub attributes: Vec<VertexAttribute>,
    pub indices: Vec<u32>,
}

impl PackedPrimitive {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

#[derive(Debug, Clone)]
pub struct PreparedPrimitive {
    pub name: String,
    pub vertices: Vec<u8>,
    pub vertex_count: usize,
    pub stride: usize,
    pub attributes: Vec<VertexAttribute>,
    pub indices: Vec<u32>,
}

impl PreparedPrimitive {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

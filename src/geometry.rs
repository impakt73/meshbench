use serde::Serialize;
use std::mem::size_of;

#[derive(Debug, Clone, Serialize)]
pub struct VertexAttribute {
    pub name: String,
    pub components: usize,
    pub element_size: usize,
}

#[derive(Debug, Clone)]
pub struct AttributeStream {
    pub name: String,
    pub components: usize,
    pub data: Vec<u8>,
}

impl AttributeStream {
    pub fn element_size(&self) -> usize {
        self.components * size_of::<f32>()
    }

    pub fn vertex_count(&self) -> usize {
        self.data
            .len()
            .checked_div(self.element_size())
            .unwrap_or(0)
    }

    pub fn info(&self) -> VertexAttribute {
        VertexAttribute {
            name: self.name.clone(),
            components: self.components,
            element_size: self.element_size(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedPrimitive {
    pub name: String,
    pub vertex_count: usize,
    pub streams: Vec<AttributeStream>,
    pub indices: Vec<u32>,
}

impl PackedPrimitive {
    pub fn position(&self) -> &AttributeStream {
        &self.streams[0]
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn attribute_info(&self) -> Vec<VertexAttribute> {
        self.streams.iter().map(AttributeStream::info).collect()
    }
}

#[derive(Debug, Clone)]
pub struct PreparedPrimitive {
    pub name: String,
    pub vertex_count: usize,
    pub streams: Vec<AttributeStream>,
    pub indices: Vec<u32>,
}

impl PreparedPrimitive {
    pub fn position(&self) -> &AttributeStream {
        &self.streams[0]
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn attribute_info(&self) -> Vec<VertexAttribute> {
        self.streams.iter().map(AttributeStream::info).collect()
    }
}

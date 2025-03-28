use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct PushConstants {
    pub strand_count: u32, // stub in case we need push constants
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct StrandMeta {
    pub offset: u32, // offset into the index buffer for this strand
    pub count: u32,  // number of vertices in this strand
}

impl From<(u32, u32)> for StrandMeta {
    fn from((offset, count): (u32, u32)) -> Self {
        Self { offset, count }
    }
}
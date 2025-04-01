use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct PushConstants {
    // pub strand_count: u32, // stub in case we need push constants
    pub workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    pub num_elements: u32,    // Generic count (e.g., num_strands or num_tiles)
    pub scan_load_base: u32,
    pub scan_save_base: u32,
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct StrandMeta {
    pub count: u32,  // number of vertices in this strand
    pub offset: u32, // offset into the index buffer for this strand
}

impl From<(u32, u32)> for StrandMeta {
    fn from((count, offset): (u32, u32)) -> Self {
        Self { count, offset }
    }
}
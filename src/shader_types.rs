use bevy::{
    math::{Vec3, bounding::Aabb3d},
    render::render_resource::ShaderType,
};
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct PushConstants {
    // pub strand_count: u32, // stub in case we need push constants
    pub workgroup_offset: u32, // For dispatch_workgroup_ext compatibility
    pub num_elements: u32,     // Generic count (e.g., num_strands or num_tiles)
    pub scan_load_base: u32,
    pub scan_save_base: u32,
    pub stochastic_cull_enabled: u32,
    pub cull_min_dist: f32,
    pub cull_max_dist: f32,
    pub cull_exponent: f32,
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct StrandMeta {
    pub count: u32,  // number of vertices in this strand
    pub offset: u32, // offset into the index buffer for this strand
    pub material_idx: u32,
    pad1: u32,
}

impl From<(u32, u32)> for StrandMeta {
    fn from((count, offset): (u32, u32)) -> Self {
        Self {
            count,
            offset,
            material_idx: 0,
            pad1: 0,
        }
    }
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct Aabb {
    pub min: Vec3,
    pad1: f32,
    pub max: Vec3,
    pad2: f32,
}

impl From<Aabb3d> for Aabb {
    fn from(aabb: Aabb3d) -> Self {
        Self {
            min: aabb.min.into(),
            max: aabb.max.into(),
            pad1: 0.0,
            pad2: 0.0,
        }
    }
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct StrandGeo {
    pub strand_count: u32,
    pub max_segments_in_strand: u32,
    pad1: u32,
    pad2: u32,
    pub aabb: Aabb,
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct FinePrepassTask {
    pub inst_id: u32,
    pub strand_local: u32,
}

#[derive(Debug, Clone, ShaderType)]
#[repr(C)]
pub struct BinningTask {
    pub id_info: u32,
    pub chunk_id: u32,
    pub seg_idx: u32,
    // bit idx[0]: flag = 0 if view 1 if light.
    // bit ids[1:2]: light type if idx[0] is 1. (point = 0, spot = 1, directional = 2)
    // Rest: index into the corresponding light buffer.
    pub packed_field: u32,
}

impl StrandGeo {
    pub fn new(strand_count: u32, max_segments_in_strand: u32, aabb: Aabb3d) -> Self {
        Self {
            strand_count,
            max_segments_in_strand,
            pad1: 0,
            pad2: 0,
            aabb: aabb.into(),
        }
    }
}

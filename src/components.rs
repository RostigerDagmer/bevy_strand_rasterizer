use bevy::{math::bounding::Aabb3d, prelude::*, render::{extract_component::ExtractComponent, storage::ShaderStorageBuffer}};
use bytemuck::{Pod, Zeroable};


#[derive(Component, ExtractComponent, Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Handle<ShaderStorageBuffer>,
    pub indices: Handle<ShaderStorageBuffer>,
    pub meta: Handle<ShaderStorageBuffer>,
    pub geos: Handle<ShaderStorageBuffer>,
    pub strand_count: u32,
    pub max_segments_in_strand: u32,
    pub aabb: Aabb3d,
}


#[derive(Component, ExtractComponent, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct FroxelConfig {
    pub screen_width: u32,
    pub screen_height: u32,
    pub froxel_size_x: u32,
    pub froxel_size_y: u32,
    pub depth_slices: u32,
}
impl Default for FroxelConfig {
    fn default() -> Self {
        Self {
            screen_width: 1920,
            screen_height: 1080,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
        }
    }
}
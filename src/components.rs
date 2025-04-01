use bevy::{prelude::*, render::{extract_component::ExtractComponent, storage::ShaderStorageBuffer}};
use bytemuck::{Pod, Zeroable};


#[derive(Component, ExtractComponent, Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Handle<ShaderStorageBuffer>,
    pub indices: Handle<ShaderStorageBuffer>,
    pub meta: Handle<ShaderStorageBuffer>,
    pub strand_count: u32,
    pub max_segments_in_strand: u32,
}


#[derive(Component, ExtractComponent, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct FroxelConfig {
    pub screen_width: u32,
    pub screen_height: u32,
    pub froxel_size_x: u32,
    pub froxel_size_y: u32,
    pub depth_slices: u32,
    pub aabb_min_x: u32,
    pub aabb_min_y: u32,
    pub aabb_min_z: f32,
    pub aabb_max_x: u32,
    pub aabb_max_y: u32,
    pub aabb_max_z: f32,
}
impl Default for FroxelConfig {
    fn default() -> Self {
        Self {
            screen_width: 1920,
            screen_height: 1080,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
            aabb_min_x: 0,
            aabb_min_y: 0,
            aabb_min_z: 0.0,
            aabb_max_x: 1920 / 8,
            aabb_max_y: 1080 / 8,
            aabb_max_z: 1.0,
        }
    }
}
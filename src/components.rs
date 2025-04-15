use bevy::{
    math::{bounding::Aabb3d, Vec4},
    prelude::*,
    render::{extract_component::ExtractComponent, render_resource::ShaderType, storage::ShaderStorageBuffer},
};
use bytemuck::{Pod, Zeroable};

use crate::dson::DsonAsset;

#[derive(Component)]
pub struct StrandAsset {
    pub handle: Handle<DsonAsset>,
}

#[derive(Component, ExtractComponent, Debug, Clone)]
pub struct StrandGeometry {
    pub vertices: Handle<ShaderStorageBuffer>,
    pub indices: Handle<ShaderStorageBuffer>,
    pub meta: Handle<ShaderStorageBuffer>,
    pub geos: Handle<ShaderStorageBuffer>,
    pub materials: Handle<ShaderStorageBuffer>,
    pub strand_count: u32,
    pub max_segments_in_strand: u32,
    pub aabb: Aabb3d,
}

#[derive(Component, ExtractComponent, ShaderType, Reflect, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct StrandMaterial {
    pub absorption_color: Vec4,
    pub specular_color: Vec4,
    pub ambient_factor: f32,
    pub ao_factor: f32,
    pub eta: f32,
    pub beta: f32,
    pub alpha: f32,
    pub shift: f32,
    pub pad1: u32,
    pub pad2: u32,
}

impl Default for StrandMaterial {
    fn default() -> Self {
        Self {
            absorption_color: Vec4::new(0.7, 0.7, 0.7, 0.4),
            specular_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            ambient_factor: 0.05,
            ao_factor: 0.1,
            eta: 1.55,
            beta: 0.5,
            alpha: 0.3,
            shift: 0.01,
            pad1: 0,
            pad2: 0,
        }
    }
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
            depth_slices: 64,
        }
    }
}

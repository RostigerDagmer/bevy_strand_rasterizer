use crate::shader_types::PushConstants;
use bevy::{
    prelude::*,
    render::extract_resource::ExtractResource,
    render::render_resource::{BindGroup, Buffer},
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Default, Clone)]
pub struct GPUPoolAllocator {
    pub chunk_pool: Option<Buffer>,
    pub free_heads: Option<Buffer>,
}

#[derive(Default, Clone)]
pub struct PoolBuffers {
    pub prepass_queue: Option<Buffer>,
    pub binning_queue: Option<Buffer>,
    pub raster_queue: Option<Buffer>,
    pub pool_allocator: GPUPoolAllocator,
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    pub instances: HashMap<Entity, StrandAssetInstance>,
    pub pool: PoolBuffers,
}

#[derive(Clone, Copy, Debug, Resource, Reflect, PartialEq, Eq, Hash)]
pub struct ComputeInvocationDims {
    pub threads_per_workgroup: u32,
    pub subgroup_size: u32,
    pub dispatch_size: (u32, u32, u32),
}

impl Default for ComputeInvocationDims {
    fn default() -> Self {
        Self {
            threads_per_workgroup: 256,
            subgroup_size: 32,
            dispatch_size: (32768, 1, 1),
        }
    }
}

#[derive(Clone, Copy, Debug, Resource, Reflect, ExtractResource)]
pub struct TileDebugSettings {
    pub enabled: bool,
    pub gain: f32,
    pub alpha: f32,
}

impl Default for TileDebugSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            gain: 8.0,
            alpha: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug, Resource, Reflect, ExtractResource)]
pub struct StochasticCullSettings {
    pub enabled: bool,
    pub target_strands_per_pixel: f32,
    pub min_keep_probability: f32,
    pub shadow_keep_probability: f32,
}

impl Default for StochasticCullSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            target_strands_per_pixel: 2.0,
            min_keep_probability: 0.02,
            shadow_keep_probability: 1.0,
        }
    }
}

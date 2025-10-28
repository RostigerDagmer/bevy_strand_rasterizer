use crate::shader_types::PushConstants;
use bevy::{prelude::*, render::render_resource::{BindGroup, Buffer}};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Default, Clone)]
pub struct GPUPoolAllocator {
    pub chunk_pool: Option<Buffer>,
    pub free_heads: Option<Buffer>
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
    pub pool: PoolBuffers
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
            dispatch_size: (65536, 1, 1),
        }
    }
}

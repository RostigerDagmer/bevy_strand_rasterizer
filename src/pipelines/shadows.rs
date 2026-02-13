use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderStages, ShaderType,
            TextureView,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};
use bevy_vsms::allocator::VirtualSurfaceRuntime;
use bevy_gpu_paging_allocator::GpuPagingAllocator;
use std::collections::HashMap;

use crate::{
    components::FroxelConfig,
    pipelines::{layouts, prepass::StrandPrepassResources, task_contract::BINNING_POOL_CHUNK_SIZE},
    plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

use super::raster::StrandRasterizerResources;

pub const NUM_DOM_SLICES: u32 = 12;

#[derive(Resource, Default)]
pub struct StrandShadowResources {
    pub dom_array_targets: Option<(TextureView, TextureView)>,
    pub light_layer_by_entity: HashMap<Entity, u32>,
    pub light_layer_by_frustum: HashMap<u32, u32>,
    pub light_entity_by_layer: Vec<Entity>,
    pub dom_vsms_proxies: HashMap<Entity, (Entity, Entity)>, // (opacity_proxy, depth_proxy)
}

#[derive(Resource)]
pub struct StrandShadowPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shadow_pipeline: Option<CachedComputePipelineId>,
    pub allocator_epoch: u64,
}

impl StrandShadowPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_shadow_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::LIGHT_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FRUSTUM_TABLE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::FROXEL_BUCKET_HEADS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::CHUNK_POOL,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::rasterizer::RASTER_WORK_QUEUE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )
    }
}

impl FromWorld for StrandShadowPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        StrandShadowPipeline {
            bind_group_layout,
            shadow_pipeline: None,
            allocator_epoch: u64::MAX,
        }
    }
}

pub fn update_strand_shadow_pipeline(
    mut pipeline: ResMut<StrandShadowPipeline>,
    allocator: Res<GpuPagingAllocator>,
    vsms_runtime: Res<VirtualSurfaceRuntime>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    if pipeline.shadow_pipeline.is_some() && pipeline.allocator_epoch == current_state {
        return;
    }

    let (Some(buffer_layout), Some(table_layout)) = (
        allocator.buffer_bind_group_layout.clone(),
        allocator.pagetable_bind_group_layout.clone(),
    ) else {
        return;
    };
    let Some(opacity_storage_layout) = vsms_runtime
        .pool_storage_bindings
        .get(&bevy_vsms::api::VirtualSurfaceKind::Opacity3D)
        .map(|b| b.layout.clone())
    else {
        return;
    };
    let Some(depth_storage_layout) = vsms_runtime
        .pool_storage_bindings
        .get(&bevy_vsms::api::VirtualSurfaceKind::Depth2DArray)
        .map(|b| b.layout.clone())
    else {
        return;
    };

    let cdefs = [
        vec![
            ShaderDefVal::UInt("MAX_TEXTURE_EXTENT".into(), MAX_TEXTURE_EXTENT),
            ShaderDefVal::UInt(
                "SIZEOF_METADATA".into(),
                std::mem::size_of::<crate::shader_types::StrandMeta>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_MATERIAL".into(),
                std::mem::size_of::<crate::components::StrandMaterial>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_GEO".into(),
                std::mem::size_of::<crate::shader_types::StrandGeo>() as u32,
            ),
            ShaderDefVal::UInt("POOL_CHUNK_SIZE".into(), BINNING_POOL_CHUNK_SIZE),
            ShaderDefVal::UInt("NUM_DOM_SLICES".into(), NUM_DOM_SLICES),
            "SHADOWS".into(),
        ],
        layouts::rasterizer::shader_defs(),
        allocator.shader_defs(),
    ]
    .concat();

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::rasterizer::RASTER_GROUP)
        .max(layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP)
        .max(layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP);
    let mut layout = vec![pipeline.bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::rasterizer::RASTER_GROUP as usize] = pipeline.bind_group_layout.clone();
    layout[layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP as usize] = opacity_storage_layout;
    layout[layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP as usize] = depth_storage_layout;

    let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");
    pipeline.shadow_pipeline = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("strand_shadow_rasterize_pipeline".into()),
        layout,
        shader: rasterize_shader,
        shader_defs: cdefs,
        push_constant_ranges: vec![PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32,
        }],
        entry_point: Some("rasterize_strands".into()),
        zero_initialize_workgroup_memory: false,
    }));
    pipeline.allocator_epoch = current_state;
}

pub fn create_strand_shadow_bind_group(
    device: &RenderDevice,
    pipeline: &StrandShadowPipeline,
    prepass_resources: &StrandPrepassResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_uniform_offset: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;

    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;
    let froxel_bucket_heads = prepass_resources.froxel_bucket_heads.as_ref().ok_or(())?;
    let chunk_pool = prepass_resources.chunk_pool.as_ref().ok_or(())?;
    let raster_work_queue = prepass_resources.raster_work_queue.as_ref().ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_shadow_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::rasterizer::VIEW_UNIFORM,
                    resource: view_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::LIGHT_UNIFORM,
                    resource: light_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::FROXEL_BUCKET_HEADS,
                    resource: froxel_bucket_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::CHUNK_POOL,
                    resource: chunk_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::rasterizer::RASTER_WORK_QUEUE,
                    resource: raster_work_queue.as_entire_binding(),
                },
            ],
        ),
        vec![view_uniform_offset.offset, view_light_uniform_offset.offset],
    ))
}

pub fn run_shadow_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadowPipeline,
    allocator: &GpuPagingAllocator,
    opacity_storage_bind_group: &BindGroup,
    depth_storage_bind_group: &BindGroup,
    froxel_config: &FroxelConfig,
    frustum_id: u32,
    resources: &StrandRasterizerResources,
    bind_group: &BindGroup,
    offsets: &[u32],
) {
    let Some(pipeline_id) = pipeline.shadow_pipeline else {
        warn!("Shadow pipeline id not ready");
        return;
    };
    let Some(shadow_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
        warn!("Shadow pipeline not found");
        return;
    };
    let Some(allocator_buffer_bind_group) = allocator.buffer_bind_group.as_ref() else {
        warn!("allocator buffer bind group is not ready yet.");
        return;
    };
    let Some(allocator_pagetable_bind_group) = allocator.pagetable_bind_group.as_ref() else {
        warn!("allocator pagetable bind group is not ready yet.");
        return;
    };

    let encoder = render_context.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Rasterize Shadow"),
        ..default()
    });
    pass.set_pipeline(shadow_pipeline);
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::rasterizer::RASTER_GROUP, bind_group, offsets);
    pass.set_bind_group(
        layouts::rasterizer::VSMS_OPACITY_WRITE_GROUP,
        opacity_storage_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::rasterizer::VSMS_DEPTH_WRITE_GROUP,
        depth_storage_bind_group,
        &[],
    );

    let pushconstants = PushConstants {
        num_elements: resources.strand_count.unwrap_or(0),
        workgroup_offset: 0,
        scan_load_base: frustum_id,
        scan_save_base: 0,
        ..Default::default()
    };
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));

    let workgroup_size_x = froxel_config.froxel_size_x;
    let workgroup_size_y = froxel_config.froxel_size_y;
    let workgroups_x = froxel_config.screen_width / workgroup_size_x;
    let workgroups_y = froxel_config.screen_height / workgroup_size_y;
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

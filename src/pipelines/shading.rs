use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            ImageSubresourceRange, PipelineCache, PushConstantRange, ShaderStages, ShaderType,
            StorageTextureAccess, Texture, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::{BindGroupBuilder, GpuPagingAllocator};
use bevy_vsms::{allocator::VirtualSurfaceRuntime, api::VirtualSurfaceKind};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::{
    pipelines::{
        layouts,
        prepass::StrandPrepassResources,
        shadows::{NUM_DOM_SLICES, StrandShadowPipeline},
    },
    plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

const MAX_SHADING_SUBSAMPLING_FACTOR: u32 = 4;
const SHADING_WORKGROUP_SIZE: u32 = 128;

#[derive(Resource, Default)]
pub struct StrandShadingResources {
    pub output_texture_resource: Option<Texture>,
    pub output_texture: Option<TextureView>,
    pub shadow_history_texture_resources: [Option<Texture>; 2],
    pub shadow_history_textures: [Option<TextureView>; 2],
    pub shadow_history_index: AtomicU32,
    pub shadow_history_needs_clear: AtomicBool,
    pub strand_count: Option<u32>,
    pub max_segments_in_strand: Option<u32>,
    pub max_strands_in_instance: Option<u32>,
    pub layer_count: Option<u32>,
}

#[derive(Resource)]
pub struct StrandShadingPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shading_pipeline: Option<CachedComputePipelineId>,
    pub allocator_epoch: u64,
    pub opacity_sample_count: u32,
    pub depth_sample_count: u32,
}

impl StrandShadingPipeline {
    pub fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor::new(
            "strand_shading_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: layouts::shading::VIEW_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::LIGHT_UNIFORM,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::BINNING_QUEUE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::FRUSTUM_TABLE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::OUTPUT_TEXTURE,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::STRAND_INSTANCES,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::SHADOW_DOM_SURFACE_IDS,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::SHADOW_HISTORY_PREV,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: bevy::render::render_resource::TextureSampleType::Float {
                            filterable: false,
                        },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: layouts::shading::SHADOW_HISTORY_NEXT,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        )
    }

    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        let descriptor = Self::bind_group_layout_descriptor();
        device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries)
    }
}

impl FromWorld for StrandShadingPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        StrandShadingPipeline {
            bind_group_layout,
            shading_pipeline: None,
            allocator_epoch: u64::MAX,
            opacity_sample_count: 0,
            depth_sample_count: 0,
        }
    }
}

pub fn update_strand_shading_pipeline(
    mut pipeline: ResMut<StrandShadingPipeline>,
    allocator: Res<GpuPagingAllocator>,
    vsms_runtime: Res<VirtualSurfaceRuntime>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    let Some(opacity_pool_binding) = vsms_runtime
        .pool_bindings
        .get(&VirtualSurfaceKind::Opacity3D)
    else {
        return;
    };
    let Some(depth_pool_binding) = vsms_runtime
        .pool_bindings
        .get(&VirtualSurfaceKind::Depth2DArray)
    else {
        return;
    };
    let opacity_sample_count = opacity_pool_binding.texture_count;
    let depth_sample_count = depth_pool_binding.texture_count;
    if pipeline.shading_pipeline.is_some()
        && pipeline.allocator_epoch == current_state
        && pipeline.opacity_sample_count == opacity_sample_count
        && pipeline.depth_sample_count == depth_sample_count
    {
        return;
    }

    if allocator.buffer_bind_group_layout.is_none()
        || allocator.pagetable_bind_group_layout.is_none()
    {
        return;
    }
    let allocator_layout_entries = allocator.layout_entries();
    let buffer_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_buffer_layout",
        &allocator_layout_entries.pools,
    );
    let table_layout = BindGroupLayoutDescriptor::new(
        "gpu_paging_allocator_table_layout",
        &allocator_layout_entries.page_tables,
    );
    let opacity_pool_layout = opacity_pool_binding.layout_descriptor.clone();
    let depth_pool_layout = depth_pool_binding.layout_descriptor.clone();

    let cdefs = [
        vec![
            ShaderDefVal::UInt("MAX_TEXTURE_EXTENT".into(), MAX_TEXTURE_EXTENT),
            ShaderDefVal::UInt("WORKGROUP_SIZE".into(), SHADING_WORKGROUP_SIZE),
            ShaderDefVal::UInt(
                "SIZEOF_METADATA".into(),
                std::mem::size_of::<crate::shader_types::StrandMeta>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_MATERIAL".into(),
                std::mem::size_of::<crate::components::StrandMaterial>() as u32,
            ),
            ShaderDefVal::UInt(
                "SIZEOF_VERTEX".into(),
                std::mem::size_of::<bevy::math::Vec4>() as u32,
            ),
            ShaderDefVal::UInt("NUM_DOM_SLICES".into(), NUM_DOM_SLICES),
            ShaderDefVal::UInt(
                layouts::shading::VSMS_OPACITY_POOL_TEXTURE_COUNT_DEF.into(),
                opacity_sample_count,
            ),
            ShaderDefVal::UInt(
                layouts::shading::VSMS_DEPTH_POOL_TEXTURE_COUNT_DEF.into(),
                depth_sample_count,
            ),
        ],
        layouts::shading::shader_defs(),
        allocator.shader_defs(),
    ]
    .concat();

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::shading::SHADING_GROUP)
        .max(layouts::shading::VSMS_OPACITY_WRITE_GROUP)
        .max(layouts::shading::VSMS_DEPTH_WRITE_GROUP)
        .max(layouts::shading::VSMS_OPACITY_TABLE_GROUP)
        .max(layouts::shading::VSMS_DEPTH_TABLE_GROUP);
    let shading_layout = StrandShadingPipeline::bind_group_layout_descriptor();
    let vsms_table_layout = StrandShadowPipeline::vsms_table_bind_group_layout_descriptor();
    let mut layout = vec![shading_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::shading::SHADING_GROUP as usize] = shading_layout;
    layout[layouts::shading::VSMS_OPACITY_WRITE_GROUP as usize] = opacity_pool_layout;
    layout[layouts::shading::VSMS_DEPTH_WRITE_GROUP as usize] = depth_pool_layout;
    layout[layouts::shading::VSMS_OPACITY_TABLE_GROUP as usize] = vsms_table_layout.clone();
    layout[layouts::shading::VSMS_DEPTH_TABLE_GROUP as usize] = vsms_table_layout;

    let shader = shader_loader.load("shaders/strand_shading.wgsl");
    pipeline.shading_pipeline = Some(pipeline_cache.queue_compute_pipeline(
        ComputePipelineDescriptor {
            label: Some("strand_shading_pipeline".into()),
            layout,
            shader,
            shader_defs: cdefs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: Some("shade_strands".into()),
            zero_initialize_workgroup_memory: false,
        },
    ));
    pipeline.allocator_epoch = current_state;
    pipeline.opacity_sample_count = opacity_sample_count;
    pipeline.depth_sample_count = depth_sample_count;
}

pub fn create_strand_shading_bind_group(
    device: &RenderDevice,
    pipeline: &StrandShadingPipeline,
    prepass_resources: &StrandPrepassResources,
    shading_resources: &StrandShadingResources,
    view_buffer: &BindingResource,
    light_buffer: &BindingResource,
    view_uniform_offset: &ViewUniformOffset,
    view_light_uniform_offset: &ViewLightsUniformOffset,
) -> Result<(BindGroup, Vec<u32>), ()> {
    let layout = &pipeline.bind_group_layout;
    let output_texture = shading_resources.output_texture.as_ref().ok_or(())?;
    let history_read_idx = shading_resources
        .shadow_history_index
        .load(Ordering::Relaxed) as usize;
    let history_write_idx = history_read_idx ^ 1usize;
    let shadow_history_prev = shading_resources.shadow_history_textures[history_read_idx]
        .as_ref()
        .ok_or(())?;
    let shadow_history_next = shading_resources.shadow_history_textures[history_write_idx]
        .as_ref()
        .ok_or(())?;
    let binning_queue = prepass_resources.binning_queue.as_ref().ok_or(())?;
    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;
    let strand_instances = prepass_resources.strand_instances.as_ref().ok_or(())?;
    let shadow_dom_surface_ids = prepass_resources
        .shadow_dom_surface_ids
        .as_ref()
        .ok_or(())?;

    Ok((
        device.create_bind_group(
            Some("strand_shading_bind_group"),
            layout,
            &[
                BindGroupEntry {
                    binding: layouts::shading::VIEW_UNIFORM,
                    resource: view_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::shading::LIGHT_UNIFORM,
                    resource: light_buffer.clone(),
                },
                BindGroupEntry {
                    binding: layouts::shading::BINNING_QUEUE,
                    resource: binning_queue.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::OUTPUT_TEXTURE,
                    resource: BindingResource::TextureView(output_texture),
                },
                BindGroupEntry {
                    binding: layouts::shading::STRAND_INSTANCES,
                    resource: strand_instances.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::SHADOW_DOM_SURFACE_IDS,
                    resource: shadow_dom_surface_ids.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::shading::SHADOW_HISTORY_PREV,
                    resource: BindingResource::TextureView(shadow_history_prev),
                },
                BindGroupEntry {
                    binding: layouts::shading::SHADOW_HISTORY_NEXT,
                    resource: BindingResource::TextureView(shadow_history_next),
                },
            ],
        ),
        vec![view_uniform_offset.offset, view_light_uniform_offset.offset],
    ))
}

pub fn create_shading_target_texture(
    device: &RenderDevice,
    layer_count: u32,
    max_strands_in_instance: u32,
    max_strand_segment_count: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_shading_output"),
        size: Extent3d {
            width: max_strand_segment_count * MAX_SHADING_SUBSAMPLING_FACTOR,
            height: max_strands_in_instance.clamp(1, MAX_TEXTURE_EXTENT),
            depth_or_array_layers: layer_count.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

pub fn create_shadow_history_texture(
    device: &RenderDevice,
    layer_count: u32,
    max_strands_in_instance: u32,
    max_strand_segment_count: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("strand_shadow_history"),
        size: Extent3d {
            width: max_strand_segment_count * MAX_SHADING_SUBSAMPLING_FACTOR,
            height: max_strands_in_instance.clamp(1, MAX_TEXTURE_EXTENT),
            depth_or_array_layers: layer_count.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

pub fn run_shading_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadingPipeline,
    prepass_resources: &StrandPrepassResources,
    resources: &StrandShadingResources,
    allocator: &GpuPagingAllocator,
    bind_group: &BindGroup,
    opacity_pool_bind_group: &BindGroup,
    depth_pool_bind_group: &BindGroup,
    opacity_table_bind_group: &BindGroup,
    depth_table_bind_group: &BindGroup,
    offsets: &[u32],
) {
    let Some(shading_pipeline_id) = pipeline.shading_pipeline else {
        warn!("Shading pipeline id not ready");
        return;
    };
    let Some(shading_pipeline) = pipeline_cache.get_compute_pipeline(shading_pipeline_id) else {
        warn!("Shading pipeline not found");
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

    let task_capacity = prepass_resources.binning_task_capacity;
    if task_capacity == 0 {
        return;
    }

    let total_workgroups = task_capacity.div_ceil(SHADING_WORKGROUP_SIZE);
    if total_workgroups == 0 {
        return;
    }
    let workgroups_x = total_workgroups.min(65_535);
    let workgroups_y = total_workgroups.div_ceil(workgroups_x);
    if workgroups_y > 65_535 {
        warn!(
            "Shading dispatch too large: workgroups=({}, {}, 1), task_capacity={}",
            workgroups_x, workgroups_y, task_capacity
        );
        return;
    }

    let encoder = render_context.command_encoder();
    if resources
        .shadow_history_needs_clear
        .swap(false, Ordering::Relaxed)
    {
        let clear_range = ImageSubresourceRange {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        };
        for texture in &resources.shadow_history_texture_resources {
            if let Some(texture) = texture.as_ref() {
                encoder.clear_texture(texture, &clear_range);
            }
        }
    }
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Strand Shading"),
        ..default()
    });
    pass.set_pipeline(shading_pipeline);
    pass.set_bind_group(allocator.buffer_group_idx, allocator_buffer_bind_group, &[]);
    pass.set_bind_group(
        allocator.table_group_idx,
        allocator_pagetable_bind_group,
        &[],
    );
    pass.set_bind_group(layouts::shading::SHADING_GROUP, bind_group, offsets);
    pass.set_bind_group(
        layouts::shading::VSMS_OPACITY_WRITE_GROUP,
        opacity_pool_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::shading::VSMS_DEPTH_WRITE_GROUP,
        depth_pool_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::shading::VSMS_OPACITY_TABLE_GROUP,
        opacity_table_bind_group,
        &[],
    );
    pass.set_bind_group(
        layouts::shading::VSMS_DEPTH_TABLE_GROUP,
        depth_table_bind_group,
        &[],
    );

    let pushconstants = PushConstants {
        num_elements: task_capacity,
        workgroup_offset: workgroups_x.saturating_mul(SHADING_WORKGROUP_SIZE),
        scan_load_base: 0,
        scan_save_base: 0,
        ..Default::default()
    };
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    resources
        .shadow_history_index
        .fetch_xor(1, Ordering::Relaxed);
}

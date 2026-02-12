use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, PushConstantRange, ShaderStages,
            ShaderType, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;

use crate::{
    pipelines::{layouts, prepass::StrandPrepassResources},
    plugin::MAX_TEXTURE_EXTENT,
    shader_types::PushConstants,
};

const MAX_SHADING_SUBSAMPLING_FACTOR: u32 = 4;
const SHADING_WORKGROUP_SIZE: u32 = 128;

#[derive(Resource, Default)]
pub struct StrandShadingResources {
    pub output_texture_resource: Option<Texture>,
    pub output_texture: Option<TextureView>,
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
}

impl StrandShadingPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
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
            ],
        )
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
        }
    }
}

pub fn update_strand_shading_pipeline(
    mut pipeline: ResMut<StrandShadingPipeline>,
    allocator: Res<GpuPagingAllocator>,
    shader_loader: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let current_state = allocator.bindgroups_epoch;
    if pipeline.shading_pipeline.is_some() && pipeline.allocator_epoch == current_state {
        return;
    }

    let (Some(buffer_layout), Some(table_layout)) = (
        allocator.buffer_bind_group_layout.clone(),
        allocator.pagetable_bind_group_layout.clone(),
    ) else {
        return;
    };

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
        ],
        layouts::shading::shader_defs(),
        allocator.shader_defs(),
    ]
    .concat();

    let max_group = allocator
        .buffer_group_idx
        .max(allocator.table_group_idx)
        .max(layouts::shading::SHADING_GROUP);
    let mut layout = vec![pipeline.bind_group_layout.clone(); (max_group + 1) as usize];
    layout[allocator.buffer_group_idx as usize] = buffer_layout;
    layout[allocator.table_group_idx as usize] = table_layout;
    layout[layouts::shading::SHADING_GROUP as usize] = pipeline.bind_group_layout.clone();

    let shader = shader_loader.load("shaders/strand_shading.wgsl");
    pipeline.shading_pipeline = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
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
    }));
    pipeline.allocator_epoch = current_state;
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
    let binning_queue = prepass_resources.binning_queue.as_ref().ok_or(())?;
    let frustum_table = prepass_resources.frustum_table.as_ref().ok_or(())?;

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

pub fn run_shading_pass(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &StrandShadingPipeline,
    prepass_resources: &StrandPrepassResources,
    resources: &StrandShadingResources,
    allocator: &GpuPagingAllocator,
    bind_group: &BindGroup,
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

    let workgroups = task_capacity.div_ceil(SHADING_WORKGROUP_SIZE);
    if workgroups == 0 {
        return;
    }

    let encoder = render_context.command_encoder();
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

    let pushconstants = PushConstants {
        num_elements: task_capacity,
        workgroup_offset: resources.max_segments_in_strand.unwrap_or(0),
        scan_load_base: 0,
        scan_save_base: 0,
        ..Default::default()
    };
    pass.set_push_constants(0, bytemuck::bytes_of(&pushconstants));
    pass.dispatch_workgroups(workgroups, 1, 1);
}

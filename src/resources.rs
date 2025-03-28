use bevy::{
    prelude::*,
    render::{render_resource::{
        BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CachedComputePipelineId, ComputePipeline, ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderDefVal, ShaderStages, StorageTextureAccess, TextureFormat, TextureView, TextureViewDimension
    }, renderer::RenderDevice},
    utils::HashMap,
};

use crate::{components::FroxelConfig, shader_types::PushConstants};

#[derive(Resource)]
pub struct StrandComputePipeline {
    // stub
    pub bind_group_layout: BindGroupLayout,
    pub rasterize_pipeline: CachedComputePipelineId,
}

impl StrandComputePipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_rasterizer_bind_group_layout",
            &[
                // Vertex buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Index buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Index buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Froxel buffer (read-write storage buffer)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Froxel configuration (uniform buffer)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // View Uniform Buffer
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )
    }
}

impl FromWorld for StrandComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let rasterize_shader = shader_loader.load("shaders/strand_rasterizer.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let rasterize_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_rasterize_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: rasterize_shader,
            shader_defs: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "rasterize_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!(
            "Created strand raster compute pipelines: rasterize={:?}", rasterize_pipeline
        );

        StrandComputePipeline {
            bind_group_layout,
            rasterize_pipeline,
        }
    }
}


// Assume you have resources for your binning pipelines and bind groups
#[derive(Resource)]
pub struct StrandBinningPipeline {
    pub count_pipeline: CachedComputePipelineId,
    // Scan pipelines (can reuse handles if shader source is shared)
    pub scan_sums_pipeline: CachedComputePipelineId,
    pub scan_last_pipeline: CachedComputePipelineId,
    pub scan_prfx_pipeline: CachedComputePipelineId,
    // Initialize placement indices pipeline (optional, could be simple)
    pub init_placement_idx_pipeline: CachedComputePipelineId,
    pub place_pipeline: CachedComputePipelineId,

    pub bind_group_layout: BindGroupLayout,
}

impl StrandBinningPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        device.create_bind_group_layout(
            "strand_binning_bind_group_layout",
            &[
                // Strand points buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Strand metadata buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile counts buffer (atomic<u32> array)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile offsets buffer (u32 array)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Packed segments buffer (SegmentRef array)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Current tile write indices buffer (atomic<u32> array)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )
    }
}

impl FromWorld for StrandBinningPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let binning_shader = shader_loader.load("shaders/binning_shad.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let cdefs = vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                super::NUMBER_OF_THREADS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt("NUMBER_OF_THREADS_PER_SUBGROUP".into(), 64), // TODO
            ShaderDefVal::UInt(
                "NUMBER_OF_ROWS_PER_WORKGROUP".into(),
                super::NUMBER_OF_ROWS_PER_WORKGROUP,
            ),
        ];

        let count_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_count_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_COUNT".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "count_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        let scan_sums_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_sums_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_SUMS".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "scan_sums".into(),
            zero_initialize_workgroup_memory: false,
        });

        let scan_last_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_last_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_LAST".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "scan_last".into(),
            zero_initialize_workgroup_memory: false,
        });

        let scan_prfx_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_prfx_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_PRFX".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "scan_prfx".into(),
            zero_initialize_workgroup_memory: false,
        });

        let init_placement_idx_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_init_placement_idx_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_INIT_PLACE".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "init_placement_idx".into(),
            zero_initialize_workgroup_memory: false,
        });

        let place_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_place_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_PLACE".into()]].concat(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "place_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!(
            "Created strand binning pipelines: count={:?}, scan_sums={:?}, scan_last={:?}, scan_prfx={:?}, init_placement_idx={:?}, place={:?}",
            count_pipeline, scan_sums_pipeline, scan_last_pipeline, scan_prfx_pipeline, init_placement_idx_pipeline, place_pipeline
        );

        StrandBinningPipeline {
            count_pipeline,
            scan_sums_pipeline,
            scan_last_pipeline,
            scan_prfx_pipeline,
            init_placement_idx_pipeline,
            place_pipeline,
            bind_group_layout
        }
    }
}

#[derive(Resource, Default)]
pub struct StrandBinningBindGroup {
    // Buffers needed across stages (might need multiple BindGroup resources)
    pub strand_points_buffer: Option<Buffer>,
    pub strand_metadata_buffer: Option<Buffer>, // {offset, count} per strand
    pub tile_counts_buffer: Option<Buffer>,     // array<atomic<u32>, num_tiles>
    pub tile_offsets_buffer: Option<Buffer>, // array<u32>, size >= num_tiles + 1 (maybe larger for scan temp)
    pub packed_segments_buffer: Option<Buffer>, // array<SegmentRef>, needs resizing based on scan total
    pub current_tile_write_indices_buffer: Option<Buffer>, // array<atomic<u32>, num_tiles>

    // Bind groups for each stage
    pub count_bind_group: Option<BindGroup>,
    pub scan_bind_group: Option<BindGroup>, // Binds tile_counts, tile_offsets, tnumber_seg (last elem of offsets)
    pub init_placement_idx_bind_group: Option<BindGroup>, // Binds tile_offsets, current_tile_write_indices
    pub place_bind_group: Option<BindGroup>, // Binds geometry, tile_offsets, current_tile_write_indices, packed_segments
}

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub bind_group: Option<BindGroup>,
    pub froxel_buffer: Option<Buffer>,
    pub froxel_config_buffer: Option<Buffer>,
    pub output_texture: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub frustrum_config: Option<FroxelConfig>,
}

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    instances: HashMap<Entity, StrandAssetInstance>,
}

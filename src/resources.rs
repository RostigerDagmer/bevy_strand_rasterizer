use bevy::{
    core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType, BlendState, Buffer,
            BufferBindingType, BufferSize, CachedComputePipelineId, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, ComputePipeline, ComputePipelineDescriptor, FilterMode,
            FragmentState, MultisampleState, PipelineCache, PrimitiveState, PushConstantRange,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal,
            ShaderStages, ShaderType, StorageTextureAccess, TextureFormat, TextureSampleType,
            TextureView, TextureViewDimension,
        },
        renderer::RenderDevice,
        view::ViewUniform,
    },
    utils::HashMap,
};

use crate::{components::FroxelConfig, shader_types::PushConstants};

#[derive(Resource)]
pub struct StrandRasterizerPipeline {
    // stub
    pub bind_group_layout: BindGroupLayout,
    pub rasterize_pipeline: CachedComputePipelineId,
}

impl StrandRasterizerPipeline {
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
                // Index Buffer
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
                // Meta buffer (read-only storage buffer)
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
                // Tile offsets buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tile counts buffer (for debug; read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Froxel buffer (read-only storage buffer)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: 6,
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
                    binding: 7,
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
                    binding: 8,
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

impl FromWorld for StrandRasterizerPipeline {
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
            "Created strand raster compute pipelines: rasterize={:?}",
            rasterize_pipeline
        );

        StrandRasterizerPipeline {
            bind_group_layout,
            rasterize_pipeline,
        }
    }
}

#[derive(Resource)]
pub struct StrandShadingPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub shading_pipeline: CachedComputePipelineId,
}

impl StrandShadingPipeline {
    pub fn create_bind_group_layout(device: &RenderDevice) -> BindGroupLayout {
        // We shade in strand space so we only need the vertex, index and meta buffers in terms of geometry.
        // We also need the View and light buffers and an output buffer containing the shading data along line segments.
        device.create_bind_group_layout(
            "strand_shading_bind_group_layout",
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
                // Index Buffer
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
                // Meta buffer (read-only storage buffer)
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
                // View Uniform Buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // Light Uniform Buffer
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output texture (write-only storage texture)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // NOTE: Additional textures here if necessary for more accurate blending during rasterization
            ],
        )
    }
}

impl FromWorld for StrandShadingPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let bind_group_layout = Self::create_bind_group_layout(device);

        let shader_loader = world.resource::<AssetServer>();
        let shading_shader = shader_loader.load("shaders/strand_shading.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let cdefs = vec![
            ShaderDefVal::UInt(
                "MAX_TEXTURE_EXTENT".into(),
                crate::MAX_TEXTURE_EXTENT, // Use crate:: constant
            ),
        ];

        let shading_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_shading_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader: shading_shader,
            shader_defs: cdefs,
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
            entry_point: "shade_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!(
            "Created strand shading compute pipelines: shading={:?}",
            shading_pipeline
        );

        StrandShadingPipeline {
            bind_group_layout,
            shading_pipeline,
        }
    }
}

// Assume you have resources for your binning pipelines and bind groups
#[derive(Resource)]
pub struct StrandBinningPipeline {
    pub count_pipeline: CachedComputePipelineId,
    pub scan_sums_pipeline: CachedComputePipelineId,
    pub scan_last_pipeline: CachedComputePipelineId,
    pub scan_prfx_pipeline: CachedComputePipelineId,
    pub init_placement_idx_pipeline: CachedComputePipelineId,
    pub place_pipeline: CachedComputePipelineId,

    // Separate layouts for each stage requiring distinct bindings
    pub count_layout: BindGroupLayout,
    pub scan_layout: BindGroupLayout,
    pub init_place_layout: BindGroupLayout,
    pub place_layout: BindGroupLayout,
}

impl StrandBinningPipeline {
    // Helper to create common buffer binding entries
    fn storage_buffer_entry(
        binding: u32,
        read_only: bool,
        size: Option<BufferSize>,
    ) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: size,
            },
            count: None,
        }
    }

    fn uniform_buffer_entry(
        binding: u32,
        has_dynamic_offset: bool,
        min_binding_size: Option<BufferSize>,
    ) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size,
            },
            count: None,
        }
    }
}

impl FromWorld for StrandBinningPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();

        // --- Define Layouts ---
        // Layout for STAGE_COUNT
        let count_layout = device.create_bind_group_layout(
            "strand_binning_count_layout",
            &[
                Self::storage_buffer_entry(0, true, None),  // vertices
                Self::storage_buffer_entry(1, true, None),  // indices
                Self::storage_buffer_entry(2, true, None),  // strand_meta
                Self::storage_buffer_entry(3, false, None), // tile_counts_buffer (atomic write)
                Self::uniform_buffer_entry(
                    4,
                    false,
                    Some(BufferSize::new(std::mem::size_of::<FroxelConfig>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(5, false, Some(ViewUniform::min_size())), // view
            ],
        );

        // Layout for STAGE_SCAN_* (Matches reference scan bindings 0, 1)
        let scan_layout = device.create_bind_group_layout(
            "strand_binning_scan_layout",
            &[
                Self::storage_buffer_entry(0, true, None), // tile_counts_buffer (read)
                Self::storage_buffer_entry(1, false, None), // tile_offsets_buffer (read/write)
                                                           // Binding 4 (tnumber_seg in ref) is implicitly handled by writing to end of tile_offsets_buffer
            ],
        );

        // Layout for STAGE_INIT_PLACE
        let init_place_layout = device.create_bind_group_layout(
            "strand_binning_init_place_layout",
            &[
                Self::storage_buffer_entry(0, true, None), // tile_offsets_buffer (read)
                Self::storage_buffer_entry(1, false, None), // current_tile_write_indices_buffer (atomic write)
            ],
        );

        // Layout for STAGE_PLACE
        let place_layout = device.create_bind_group_layout(
            "strand_binning_place_layout",
            &[
                Self::storage_buffer_entry(0, true, None),  // vertices
                Self::storage_buffer_entry(1, true, None),  // indices
                Self::storage_buffer_entry(2, true, None),  // strand_meta
                Self::storage_buffer_entry(3, false, None), // current_tile_write_indices_buffer (atomic read/write)
                Self::storage_buffer_entry(4, false, None), // packed_segments_buffer (write)
                Self::uniform_buffer_entry(
                    5,
                    false,
                    Some(BufferSize::new(std::mem::size_of::<FroxelConfig>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(6, false, Some(ViewUniform::min_size())), // view
            ],
        );

        // --- Queue Pipelines ---
        let shader_loader = world.resource::<AssetServer>();
        let binning_shader = shader_loader.load("shaders/strand_binning.wgsl"); // Changed name
        let pipeline_cache = world.resource::<PipelineCache>();
        // let subgroup_size = world.resource::<bevy_radix_sort::get_subgroup_size::SubgroupSize>(); // Get subgroup size if needed
        let subgroup_size: (u32, u32) = (32, 32); // Defaults because SubgroupSize plugin doesn't work at this stage of app-build.

        // Shader defs for scan stages (match reference)
        let cdefs = vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                crate::NUMBER_OF_THREADS_PER_WORKGROUP, // Use crate:: constant
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
                subgroup_size.0, // Use detected subgroup size
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_ROWS_PER_WORKGROUP".into(), // This might not be relevant if scan logic is adapted
                crate::NUMBER_OF_ROWS_PER_WORKGROUP,
            ),
        ];

        // Push constant range covering the potentially larger struct
        let push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32, // Make sure PushConstants is large enough
        };

        let count_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_count_pipeline".into()),
            layout: vec![count_layout.clone()], // Use specific layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_COUNT".into()]].concat(), // Only define STAGE_COUNT
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: "count_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        let scan_sums_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_sums_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_SUMS".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: "scan_sums".into(),
            zero_initialize_workgroup_memory: true, // Scan often uses workgroup memory
        });

        let scan_last_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_last_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_LAST".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: "scan_last".into(),
            zero_initialize_workgroup_memory: true,
        });

        let scan_prfx_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_prfx_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_PRFX".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: "scan_prfx".into(),
            zero_initialize_workgroup_memory: true,
        });

        let init_placement_idx_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("strand_binning_init_placement_idx_pipeline".into()),
                layout: vec![init_place_layout.clone()], // Use init_place layout
                shader: binning_shader.clone(),
                // shader_defs: vec!["STAGE_INIT_PLACE".into()],
                shader_defs: [cdefs.as_slice(), &["STAGE_INIT_PLACE".into()]].concat(),
                push_constant_ranges: vec![push_constant_range.clone()], // Might not need push constants?
                entry_point: "init_placement_idx".into(),
                zero_initialize_workgroup_memory: false,
            });

        let place_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_place_pipeline".into()),
            layout: vec![place_layout.clone()], // Use place layout
            shader: binning_shader.clone(),
            // shader_defs: vec!["STAGE_PLACE".into()],
            shader_defs: [cdefs.as_slice(), &["STAGE_PLACE".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: "place_strands".into(),
            zero_initialize_workgroup_memory: false,
        });

        info!("Created strand binning pipelines");

        StrandBinningPipeline {
            count_pipeline,
            scan_sums_pipeline,
            scan_last_pipeline,
            scan_prfx_pipeline,
            init_placement_idx_pipeline,
            place_pipeline,
            count_layout,
            scan_layout,
            init_place_layout,
            place_layout,
        }
    }
}

pub struct StrandBinningBindGroup {
    // Bind groups for each stage
    pub count_bind_group: BindGroup,
    pub scan_bind_group: BindGroup, // Binds tile_counts, tile_offsets, tnumber_seg (last elem of offsets)
    pub init_placement_idx_bind_group: BindGroup, // Binds tile_offsets, current_tile_write_indices
    pub place_bind_group: BindGroup, // Binds geometry, tile_offsets, current_tile_write_indices, packed_segments
}

#[derive(Resource, Default)]
pub struct StrandBinningBuffers {
    // Keep Option<Buffer> for tile_counts, offsets, etc.
    pub tile_counts_buffer: Option<Buffer>,
    pub tile_offsets_buffer: Option<Buffer>,
    pub current_tile_write_indices_buffer: Option<Buffer>,
    pub packed_segments_buffer: Option<Buffer>,
    // Add handles/references needed from StrandGeometry
    pub vertex_buffer: Option<Buffer>, // Store the actual buffer ref
    pub index_buffer: Option<Buffer>,  // Store the actual buffer ref
    pub meta_buffer: Option<Buffer>,   // Store the actual buffer ref
}

#[derive(Resource, Default)]
pub struct StrandRasterizerResources {
    pub pipeline: Option<ComputePipeline>,
    pub froxel_buffer: Option<Buffer>,
    pub froxel_config_buffer: Option<Buffer>,
    pub output_texture: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub frustrum_config: Option<FroxelConfig>,
}

#[derive(Resource, Default)]
pub struct StrandShadingResources {
    pub output_texture: Option<TextureView>,
    pub strand_count: Option<u32>,
    pub max_segments_in_strand: Option<u32>,
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

#[derive(Resource)]
pub struct CompositionPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedRenderPipelineId, // Use RenderPipeline for fullscreen quad/triangle
    pub sampler: Sampler,
}

impl FromWorld for CompositionPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let layout = render_device.create_bind_group_layout(
            "composition_layout",
            &[
                // Input Scene Texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true }, // Assuming HDR intermediate
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Strand Rasterizer Output Texture
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true }, // Or Uint/Sint if format is different, but sampling rgba8unorm as float is fine
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("composition_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/strand_composite.wgsl");

        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("composition_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    // IMPORTANT: This format must match the ViewTarget format
                    // Usually HDR first, then tonemapped. Let's assume HDR for now.
                    format: TextureFormat::bevy_default(), // Use bevy's default HDR format
                    blend: Some(BlendState::ALPHA_BLENDING), // Use alpha blending
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(), // Triangle list covering screen
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        CompositionPipeline {
            layout,
            pipeline,
            sampler,
        }
    }
}

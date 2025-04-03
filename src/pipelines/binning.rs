use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType, BufferDescriptor, BufferSize, BufferUsages, CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, FilterMode, FragmentState, MultisampleState, PipelineCache, PrimitiveState, PushConstantRange, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages, ShaderType, StorageTextureAccess, TextureFormat, TextureSampleType, TextureView, TextureViewDimension
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniform,
    },
};
use bevy_radix_sort::dispatch_workgroup_ext;

use crate::{components::FroxelConfig, pipelines::layouts, shader_types::PushConstants};

use super::raster::StrandRasterizerResources;

/// The row size of the `keys` processed by each workgroup.
pub const NUMBER_OF_ROWS_PER_WORKGROUP: u32 = 16;
pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

/// The number of keys processed by this workgroup.
const SCAN_NUMBER_OF_KEYS_OFFSET: u32 = 4;
/// The scan step reads from the `number_segs` buffer starting at this index.
const SCAN_LOAD_BASE_OFFSET: u32 = 8;
/// The scan step writes to the `number_segs` buffer starting at this index.
const SCAN_SAVE_BASE_OFFSET: u32 = 12;

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
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub meta_buffer: Option<Buffer>, 
    pub geos_buffer: Option<Buffer>,
}

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
                Self::storage_buffer_entry(layouts::binning::VERTEX_BUFFER, true, None), // vertices
                Self::storage_buffer_entry(layouts::binning::INDEX_BUFFER, true, None),  // indices
                Self::storage_buffer_entry(layouts::binning::META_BUFFER, true, None), // strand_meta
                Self::storage_buffer_entry(layouts::binning::TILE_COUNTS_BUFFER, false, None), // tile_counts_buffer (atomic write)
                Self::uniform_buffer_entry(
                    layouts::binning::FROXEL_CONFIG,
                    false,
                    Some(BufferSize::new(std::mem::size_of::<FroxelConfig>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(
                    layouts::binning::VIEW_UNIFORM,
                    false,
                    Some(ViewUniform::min_size()),
                ), // view
                Self::storage_buffer_entry(layouts::binning::GEO_BUFFER, true, None), // geos
            ],
        );

        // Layout for STAGE_SCAN_* (Matches reference scan bindings 0, 1)
        let scan_layout = device.create_bind_group_layout(
            "strand_binning_scan_layout",
            &[
                Self::storage_buffer_entry(layouts::binning::TILE_COUNTS_BUFFER, true, None), // tile_counts_buffer (read)
                Self::storage_buffer_entry(layouts::binning::TILE_OFFSETS_BUFFER, false, None), // tile_offsets_buffer (read/write)
            ],
        );

        // Layout for STAGE_INIT_PLACE
        let init_place_layout = device.create_bind_group_layout(
            "strand_binning_init_place_layout",
            &[
                Self::storage_buffer_entry(layouts::binning::TILE_OFFSETS_BUFFER, true, None), // tile_offsets_buffer (read)
                Self::storage_buffer_entry(
                    layouts::binning::CURRENT_TILE_WRITE_INDICES,
                    false,
                    None,
                ), // current_tile_write_indices_buffer (atomic write)
            ],
        );

        // Layout for STAGE_PLACE
        let place_layout = device.create_bind_group_layout(
            "strand_binning_place_layout",
            &[
                Self::storage_buffer_entry(layouts::binning::VERTEX_BUFFER, true, None), // vertices
                Self::storage_buffer_entry(layouts::binning::INDEX_BUFFER, true, None),  // indices
                Self::storage_buffer_entry(layouts::binning::META_BUFFER, true, None), // strand_meta
                Self::storage_buffer_entry(
                    layouts::binning::CURRENT_TILE_WRITE_INDICES,
                    false,
                    None,
                ), // current_tile_write_indices_buffer (atomic read/write)
                Self::storage_buffer_entry(layouts::binning::FROXEL_TILE_BUFFER, false, None), // packed_segments_buffer (write)
                Self::uniform_buffer_entry(
                    layouts::binning::FROXEL_CONFIG,
                    false,
                    Some(BufferSize::new(std::mem::size_of::<FroxelConfig>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(
                    layouts::binning::VIEW_UNIFORM,
                    false,
                    Some(ViewUniform::min_size()),
                ), // view
                Self::storage_buffer_entry(layouts::binning::GEO_BUFFER, true, None), // geos
            ],
        );

        // --- Queue Pipelines ---
        let shader_loader = world.resource::<AssetServer>();
        let binning_shader = shader_loader.load("shaders/strand_binning.wgsl"); // Changed name
        let pipeline_cache = world.resource::<PipelineCache>();
        // let subgroup_size = world.resource::<bevy_radix_sort::get_subgroup_size::SubgroupSize>(); // Get subgroup size if needed
        let subgroup_size: (u32, u32) = (32, 32); // Defaults because SubgroupSize plugin doesn't work at this stage of app-build.

        // Shader defs for scan stages (match reference)
        let cdefs = [vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                NUMBER_OF_THREADS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_SUBGROUP".into(),
                subgroup_size.0, // Use detected subgroup size
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_ROWS_PER_WORKGROUP".into(), // This might not be relevant if scan logic is adapted
                NUMBER_OF_ROWS_PER_WORKGROUP,
            ),
        ] , layouts::binning::shader_defs()].concat();

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

pub fn prepare_binning_buffers(
    render_device: &RenderDevice,
    froxel_config: &FroxelConfig,
) -> (Buffer, Buffer, Buffer, Buffer) {
    let tile_size_x = froxel_config.froxel_size_x;
    let tile_size_y = froxel_config.froxel_size_y;
    let depth_slices = froxel_config.depth_slices;
    let screen_width = froxel_config.screen_width;

    let num_tiles_x = (screen_width + tile_size_x - 1) / tile_size_x;
    let num_tiles_y = (froxel_config.screen_height + tile_size_y - 1) / tile_size_y;
    let num_tiles = num_tiles_x * num_tiles_y * depth_slices;

    // Create the tile counts buffer
    let tile_counts_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_tile_counts_buffer"),
        size: num_tiles as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // initializes with zeros
    let tile_counts = vec![0u32; num_tiles as usize];
    let mut mapped = tile_counts_buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&tile_counts));
    drop(mapped);
    tile_counts_buffer.unmap();

    // Create the tile offsets buffer
    let tile_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_tile_offsets_buffer"),
        size: (num_tiles * 2 + 1) as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // initializes with zeros
    let tile_offsets = vec![0u32; (num_tiles * 2 + 1) as usize];
    let mut mapped = tile_offsets_buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&tile_offsets));
    drop(mapped);
    tile_offsets_buffer.unmap();

    // Create the current tile write indices buffer
    let current_tile_write_indices_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_current_tile_write_indices_buffer"),
        size: num_tiles as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // initializes with zeros
    let current_tile_write_indices = vec![0u32; num_tiles as usize];
    let mut mapped = current_tile_write_indices_buffer
        .slice(..)
        .get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::cast_slice(&current_tile_write_indices));
    drop(mapped);
    current_tile_write_indices_buffer.unmap();

    // Create the packed segments buffer
    let packed_segments_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("strand_packed_segments_buffer"),
        size: 1024 * 1024 * 8 * 4, // Initial size, will be resized after scan
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (
        tile_counts_buffer,
        tile_offsets_buffer,
        current_tile_write_indices_buffer,
        packed_segments_buffer,
    )
}

pub fn create_strand_binning_bind_group(
    device: &RenderDevice,
    pipeline: &StrandBinningPipeline,
    view_uniforms: BindingResource,
    raster_resources: &StrandRasterizerResources,
    binning_resources: &StrandBinningBuffers,
) -> Result<StrandBinningBindGroup, ()> {

    let strand_points_buffer = binning_resources.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = binning_resources.index_buffer.as_ref().ok_or(())?;
    let strand_metadata_buffer = binning_resources.meta_buffer.as_ref().ok_or(())?;
    // Count Bind Group
    let count_bind_group = device.create_bind_group(
        Some("strand_count_bind_group"),
        &pipeline.count_layout, // Use count_layout
        &[
            BindGroupEntry {
                binding: layouts::binning::VERTEX_BUFFER,
                resource: strand_points_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::INDEX_BUFFER,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::META_BUFFER,
                resource: strand_metadata_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::TILE_COUNTS_BUFFER,
                resource: binning_resources
                    .tile_counts_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_CONFIG,
                resource: raster_resources
                    .froxel_config_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::VIEW_UNIFORM,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: layouts::binning::GEO_BUFFER,
                resource: binning_resources
                    .geos_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
        ],
    );

    // Scan Bind Group
    let scan_bind_group = device.create_bind_group(
        Some("strand_scan_bind_group"),
        &pipeline.scan_layout, // Use scan_layout
        &[
            BindGroupEntry {
                binding: layouts::binning::TILE_COUNTS_BUFFER,
                resource: binning_resources
                    .tile_counts_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::TILE_OFFSETS_BUFFER,
                resource: binning_resources
                    .tile_offsets_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            // Implicitly handles total count via buffer structure
        ],
    );

    // Init Placement Index Bind Group
    let init_placement_idx_bind_group = device.create_bind_group(
        Some("strand_init_placement_idx_bind_group"),
        &pipeline.init_place_layout, // Use init_place_layout
        &[
            BindGroupEntry {
                binding: layouts::binning::TILE_OFFSETS_BUFFER,
                resource: binning_resources
                    .tile_offsets_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::CURRENT_TILE_WRITE_INDICES,
                resource: binning_resources
                    .current_tile_write_indices_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
        ],
    );

    // Place Bind Group
    let place_bind_group = device.create_bind_group(
        Some("strand_place_bind_group"),
        &pipeline.place_layout, // Use place_layout
        &[
            BindGroupEntry {
                binding: layouts::binning::VERTEX_BUFFER,
                resource: strand_points_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::INDEX_BUFFER,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::META_BUFFER,
                resource: strand_metadata_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::CURRENT_TILE_WRITE_INDICES,
                resource: binning_resources
                    .current_tile_write_indices_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_TILE_BUFFER,
                resource: binning_resources
                    .packed_segments_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_CONFIG,
                resource: raster_resources
                    .froxel_config_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::VIEW_UNIFORM,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: layouts::binning::GEO_BUFFER,
                resource: binning_resources
                    .geos_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            }
        ],
    );
    return Ok(StrandBinningBindGroup {
        count_bind_group,
        scan_bind_group,
        init_placement_idx_bind_group,
        place_bind_group,
    });
}


pub fn run_binning_pass(
    render_device: &RenderDevice,
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    bind_groups: &StrandBinningBindGroup,
    buffers: &StrandBinningBuffers,
    pipelines: &StrandBinningPipeline,
    froxel_config: &FroxelConfig,
    num_strands_or_segments: u32,
) {
    let encoder = render_context.command_encoder(); // Get CommandEncoder

    let tile_size_x = froxel_config.froxel_size_x;
    let tile_size_y = froxel_config.froxel_size_y;
    let depth_slices = froxel_config.depth_slices;
    let screen_width = froxel_config.screen_width;

    let num_tiles_x = (screen_width + tile_size_x - 1) / tile_size_x;
    let num_tiles_y = (froxel_config.screen_height + tile_size_y - 1) / tile_size_y;
    let num_tiles = num_tiles_x * num_tiles_y * depth_slices;

    let max_compute_workgroups_per_dimension =
        render_device.limits().max_compute_workgroups_per_dimension;

    info!(
        "max_compute_workgroups_per_dimension: {:?}",
        max_compute_workgroups_per_dimension
    );
    let threads_per_workgroup = NUMBER_OF_THREADS_PER_WORKGROUP;

    // --- Clear count buffer (important!) ---
    // Use encoder.clear_buffer(...) or a small compute shader pass
    encoder.clear_buffer(buffers.tile_counts_buffer.as_ref().unwrap(), 0, None); // Clear whole buffer

    // --- Pass 1: Count ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Count"),
            ..default()
        });
        let Some(count_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.count_pipeline)
        else {
            warn!("Count pipeline not found");
            return;
        };
        pass.set_pipeline(count_pipeline);
        pass.set_bind_group(0, &bind_groups.count_bind_group, &[]);
        // Set push constants if needed (e.g., num_strands_or_segments)
        // pass.set_push_constants(...);

        // Dispatch based on number of strands or segments
        let workgroup_size_x = 64; // Match shader
        let num_workgroups_x = (num_strands_or_segments + workgroup_size_x - 1) / workgroup_size_x;
        pass.dispatch_workgroups(num_workgroups_x, 1, 1); // Adjust dispatch logic as needed
    }

    // --- Pass 2: Scan (Adapted from reference) ---
    let total_count_buffer_offset = num_tiles * 4; // Offset to the last u32 element storing the total
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Scan"),
            ..default()
        });
        pass.set_bind_group(0, &bind_groups.scan_bind_group, &[]); // Scan bind group

        // Reuse scan pipeline handles from StrandBinningPipeline
        let Some(scan_sums_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_sums_pipeline)
        else {
            warn!("Scan sums pipeline not found");
            return;
        };
        let Some(scan_last_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_last_pipeline)
        else {
            warn!("Scan last pipeline not found");
            return;
        };
        let Some(scan_prfx_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.scan_prfx_pipeline)
        else {
            warn!("Scan prefix pipeline not found");
            return;
        };

        // Calculate scan hierarchy (copy/adapt logic from reference `run` function)
        let mut load_base = 0u32;
        // Where hierarchical sums are stored. Might be within tile_offsets_buffer itself after num_tiles+1 elements
        // Or could be a separate temp buffer included in the scan_bind_group.
        // Assuming reuse of tile_offsets_buffer for simplicity here, needs careful size calculation.
        let mut save_base = num_tiles; // Start writing sums after the main counts
        let mut rounds = vec![];
        while save_base - load_base > NUMBER_OF_THREADS_PER_WORKGROUP {
            let number_of_workgroups =
                (save_base - load_base).div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);
            rounds.push((load_base, save_base, number_of_workgroups));
            load_base = save_base;
            save_base += number_of_workgroups;
        }
        info!("Scan rounds: {:?}", rounds);

        // Scan Sums (Hierarchical reduction)
        pass.set_pipeline(scan_sums_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter() {
            // Set push constants for scan load/save base (offsets in tile_offsets_buffer)
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));
            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                0, // Handle large dispatches if needed
            );
        }

        // Scan Last (Scan the final reduced block)
        pass.set_pipeline(scan_last_pipeline);
        // info!("Scan last: load_base: {}, save_base: {}", load_base, save_base);
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));
        // Set push constant for total count offset if separate buffer not used
        pass.set_push_constants(0, bytemuck::bytes_of(&total_count_buffer_offset));
        pass.dispatch_workgroups(1, 1, 1);

        // Scan Prefix (Propagate scan results back down)
        pass.set_pipeline(scan_prfx_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
            // info!("Scan prefix: load_base: {}, save_base: {}", save_base, load_base);
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(save_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(load_base));
            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                0,
            );
        }
    }

    // --- (Optional) Read back total count for buffer resizing ---
    // let total_segment_refs = read_buffer_value_u32(..., &bind_groups.tile_offsets_buffer, total_count_buffer_offset);
    // resize_packed_segments_buffer_if_needed(..., bind_groups.packed_segments_buffer, total_segment_refs);
    // Update bind groups if buffer was recreated

    // --- Pass 2.5: Initialize Placement Indices ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Init Placement Indices"),
            ..default()
        });
        let Some(init_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.init_placement_idx_pipeline)
        else {
            warn!("Init placement indices pipeline not found");
            return;
        };
        pass.set_pipeline(init_pipeline);
        pass.set_bind_group(0, &bind_groups.init_placement_idx_bind_group, &[]);
        // Dispatch one thread per tile
        let workgroup_size = 256; // Example
        let num_workgroups = (num_tiles + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // --- Pass 3: Place ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Place"),
            ..default()
        });
        let Some(place_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.place_pipeline)
        else {
            warn!("Place pipeline not found");
            return;
        };
        pass.set_pipeline(place_pipeline);
        pass.set_bind_group(0, &bind_groups.place_bind_group, &[]);
        // Set push constants if needed
        // pass.set_push_constants(...);

        // Dispatch based on number of strands or segments (same as Pass 1)
        let workgroup_size_x = 64; // Match shader
        let num_workgroups_x = (num_strands_or_segments + workgroup_size_x - 1) / workgroup_size_x;
        pass.dispatch_workgroups(num_workgroups_x, 1, 1);
    }

    // --- Binning complete ---
    // packed_segments_buffer and tile_offsets_buffer are ready for the rasterizer
}
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferSize, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, PushConstantRange, ShaderStages, ShaderType,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniform, ViewUniformOffset},
    },
    shader::ShaderDefVal,
};

use std::collections::HashMap;

use crate::{
    components::{FroxelCapacity, FroxelConfig, TieFroxelsToView},
    pipelines::{ext::dispatch_workgroup_ext, layouts},
    shader_types::PushConstants,
};

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
    pub offsets: Vec<u32>,           // Uniform dynamic offsets
}

pub struct StrandBinningArtifactBuffers {
    pub tile_counts_buffer: Buffer,
    pub tile_offsets_buffer: Buffer,
    pub current_tile_write_indices_buffer: Buffer,
    pub packed_segments_buffer: Buffer,
    // pub indirect_buffer: Buffer,
}

#[derive(Resource, Default)]
pub struct StrandBinningBuffers {
    // Keep Option<Buffer> for tile_counts, offsets, etc.
    // pub tile_counts_buffer: HashMap<Entity, Buffer>,
    // pub tile_offsets_buffer: HashMap<Entity, Buffer>,
    // pub current_tile_write_indices_buffer: HashMap<Entity, Buffer>,
    // pub packed_segments_buffer: HashMap<Entity, Buffer>,
    pub artifacts: HashMap<Entity, StrandBinningArtifactBuffers>, // For storing artifacts per entity

    // Add handles/references needed from StrandGeometry
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub meta_buffer: Option<Buffer>,
    pub geos_buffer: Option<Buffer>,
    // pub capacity: FroxelCapacity,
}

#[derive(Resource)]
pub struct StrandBinningPipeline {
    pub count_pipeline: CachedComputePipelineId,
    pub count_pipeline_shadows: CachedComputePipelineId,
    pub scan_sums_pipeline: CachedComputePipelineId,
    pub scan_last_pipeline: CachedComputePipelineId,
    pub scan_prfx_pipeline: CachedComputePipelineId,
    pub init_placement_idx_pipeline: CachedComputePipelineId,
    pub place_pipeline: CachedComputePipelineId,
    pub place_pipeline_shadows: CachedComputePipelineId,

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
                    Some(BufferSize::new(std::mem::size_of::<[u32; 8]>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(
                    layouts::binning::VIEW_UNIFORM,
                    true,
                    Some(ViewUniform::min_size()),
                ), // view
                Self::uniform_buffer_entry(layouts::binning::LIGHT_UNIFORM, true, None),
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
                    Some(BufferSize::new(std::mem::size_of::<[u32; 8]>() as u64).unwrap()),
                ), // config
                Self::uniform_buffer_entry(
                    layouts::binning::VIEW_UNIFORM,
                    true,
                    Some(ViewUniform::min_size()),
                ), // view
                Self::uniform_buffer_entry(layouts::binning::LIGHT_UNIFORM, true, None),
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
        let cdefs = [
            vec![
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
            ],
            layouts::binning::shader_defs(),
        ]
        .concat();

        // Push constant range covering the potentially larger struct
        let push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32, // Make sure PushConstants is large enough
        };

        let count_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_count_pipeline".into()),
            layout: vec![count_layout.clone()], // Use specific layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_COUNT".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: Some("count_strands".into()),
            zero_initialize_workgroup_memory: false,
        });

        let count_pipeline_shadows =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("strand_binning_count_pipeline_shadows".into()),
                layout: vec![count_layout.clone()], // Use specific layout
                shader: binning_shader.clone(),
                shader_defs: [cdefs.as_slice(), &["STAGE_COUNT".into(), "SHADOWS".into()]].concat(),
                push_constant_ranges: vec![push_constant_range.clone()],
                entry_point: Some("count_strands".into()),
                zero_initialize_workgroup_memory: false,
            });

        let scan_sums_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_sums_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_SUMS".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: Some("scan_sums".into()),
            zero_initialize_workgroup_memory: true, // Scan often uses workgroup memory
        });

        let scan_last_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_last_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_LAST".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: Some("scan_last".into()),
            zero_initialize_workgroup_memory: true,
        });

        let scan_prfx_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_scan_prfx_pipeline".into()),
            layout: vec![scan_layout.clone()], // Use scan layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_SCAN_PRFX".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: Some("scan_prfx".into()),
            zero_initialize_workgroup_memory: true,
        });

        let init_placement_idx_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("strand_binning_init_placement_idx_pipeline".into()),
                layout: vec![init_place_layout.clone()], // Use init_place layout
                shader: binning_shader.clone(),
                shader_defs: [cdefs.as_slice(), &["STAGE_INIT_PLACE".into()]].concat(),
                push_constant_ranges: vec![push_constant_range.clone()], // Might not need push constants?
                entry_point: Some("init_placement_idx".into()),
                zero_initialize_workgroup_memory: false,
            });

        let place_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("strand_binning_place_pipeline".into()),
            layout: vec![place_layout.clone()], // Use place layout
            shader: binning_shader.clone(),
            shader_defs: [cdefs.as_slice(), &["STAGE_PLACE".into()]].concat(),
            push_constant_ranges: vec![push_constant_range.clone()],
            entry_point: Some("place_strands".into()),
            zero_initialize_workgroup_memory: false,
        });

        let place_pipeline_shadows =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("strand_binning_place_pipeline_shadows".into()),
                layout: vec![place_layout.clone()], // Use place layout
                shader: binning_shader.clone(),
                shader_defs: [cdefs.as_slice(), &["STAGE_PLACE".into(), "SHADOWS".into()]].concat(),
                push_constant_ranges: vec![push_constant_range.clone()],
                entry_point: Some("place_strands".into()),
                zero_initialize_workgroup_memory: false,
            });

        debug!("Created strand binning pipelines");

        StrandBinningPipeline {
            count_pipeline,
            count_pipeline_shadows,
            scan_sums_pipeline,
            scan_last_pipeline,
            scan_prfx_pipeline,
            init_placement_idx_pipeline,
            place_pipeline,
            place_pipeline_shadows,
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
        size: (froxel_config.screen_width as u64)
            * (froxel_config.screen_height as u64)
            * (froxel_config.depth_slices as u64)
            * 4, // Initial size, will be resized after scan
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

fn compute_active_raster_size(view_px: UVec2, tie: TieFroxelsToView) -> UVec2 {
    match tie {
        TieFroxelsToView::Native => view_px,
        TieFroxelsToView::Scaled(s) => (view_px.as_vec2() * s).floor().as_uvec2().max(UVec2::ONE),
        TieFroxelsToView::Fixed(sz) => sz.max(UVec2::ONE),
    }
}

fn tiles_for(size: UVec2, tile: UVec2) -> UVec2 {
    UVec2::new(
        (size.x + tile.x - 1) / tile.x,
        (size.y + tile.y - 1) / tile.y,
    )
}

fn grow_to_capacity(active: UVec2, cap: &mut FroxelCapacity, depth_slices: u32) -> bool {
    let mut need_realloc = false;

    let target_cap_x = active.x.next_power_of_two().max(8); // clamp to reasonable min
    let target_cap_y = active.y.next_power_of_two().max(8);

    if target_cap_x > cap.tiles_cap_x
        || target_cap_y > cap.tiles_cap_y
        || depth_slices > cap.depth_cap
    {
        cap.tiles_cap_x = target_cap_x;
        cap.tiles_cap_y = target_cap_y;
        cap.depth_cap = depth_slices;
        cap.tiles_capacity =
            (cap.tiles_cap_x as u64) * (cap.tiles_cap_y as u64) * (cap.depth_cap as u64);
        need_realloc = true;
    }
    need_realloc
}

// fn prepare_resize_and_capacity(
//     render_device: Res<RenderDevice>,
//     render_queue: Res<bevy::render::renderer::RenderQueue>,
//     pipeline_cache: Res<PipelineCache>,
//     views: Query<(&ExtractedView, Option<&TieFroxelsToView>)>, // or your camera query
//     mut raster_resources: ResMut<StrandRasterizerResources>,
//     mut binning_resources: ResMut<StrandBinningBuffers>,
// ) {
//     for (entity, per_view) in raster_resources.froxel_config_buffer.clone().into_iter() {
//         // Get view size
//         let Ok((extracted, tie_mode)) = views.get(entity) else {
//             warn!("No view on entity: {:?}", entity);
//             continue;
//         };
//         let extracted = extracted;
//         let view_px = UVec2::new(extracted.viewport.z as u32, extracted.viewport.w as u32);
//         let tie = per_view.and_then(|_| raster_resources.tie_modes.get(&entity))
//                           .copied()
//                           .unwrap_or(TieFroxelsToView::Native);

//         // Compute active raster size & tiles
//         let internal_px = compute_active_raster_size(view_px, tie);
//         let config = raster_resources.frustrum_config[&entity];
//         let tile = UVec2::new(
//             config.froxel_size_x,
//             config.froxel_size_y,
//         );
//         let tiles_active = tiles_for(internal_px, tile);
//         let num_tiles_total = tiles_active.x as u64 * tiles_active.y as u64
//                             * raster_resources.frustrum_config[&entity].depth_slices as u64;

//         // Update uniform (ACTIVE values only)
//         let mut cfg = raster_resources.frustrum_config.get_mut(&entity).unwrap().clone();
//         cfg.screen_width  = internal_px.x;
//         cfg.screen_height = internal_px.y;
//         cfg.num_tiles_x   = tiles_active.x;
//         cfg.num_tiles_y   = tiles_active.y;
//         cfg.num_tiles_total = num_tiles_total;

//         let cfg_buf = raster_resources.froxel_config_buffer.get(&entity).unwrap();
//         render_queue.write_buffer(cfg_buf, 0, bytemuck::bytes_of(&cfg));

//         // Capacity check
//         let artifacts = binning_resources.artifacts.get_mut(&entity).unwrap();
//         let mut cap = artifacts.capacity.clone();
//         let need_realloc = grow_to_capacity(tiles_active, &mut cap, cfg.depth_slices);

//         // Packed buffer capacity policy (optionally also check a soft limit)
//         // cap.packed_capacity_bytes = compute_or_grow_packed_capacity(&cap, cfg);

//         if need_realloc {
//             // Recreate storage buffers sized to cap.tiles_capacity
//             let new_artifacts = recreate_binning_buffers_with_capacity(
//                 &render_device,
//                 &cap,
//                 /* initial_zero = */ true,
//             );

//             // Recreate or grow render target textures to cover internal_px
//             let (target_view, depth_view) =
//                 recreate_render_target_texture(&render_device, &config);

//             // Swap in artifacts + views
//             *artifacts = new_artifacts;
//             raster_resources.output_texture = Some(target_view);
//             raster_resources.output_depth   = Some(depth_view);

//             // Recreate bind groups for this entity (buffers changed)
//             let bg = create_strand_binning_bind_group(
//                 &entity,
//                 &render_device,
//                 &raster_resources.binning_pipeline, // wherever you keep it
//                 /* view uniforms... */,
//                 /* light uniforms... */,
//                 &raster_resources,
//                 &binning_resources,
//             ).expect("bind groups");

//             raster_resources.binning_bind_groups.insert(entity, bg);
//         } else {
//             // No rebinding: just ensure buffers are cleared per-frame before use (you already clear tile_counts in the pass)
//             // If you need to clear offsets/current indices here, do it via compute or encoder.clear_buffer
//         }
//     }
// }

pub fn create_strand_binning_bind_group(
    entity: &Entity,
    device: &RenderDevice,
    pipeline: &StrandBinningPipeline,
    view_uniforms: BindingResource,
    view_uniform_offset: &ViewUniformOffset,
    light_uniform: BindingResource,
    light_uniform_offset: &ViewLightsUniformOffset,
    raster_resources: &StrandRasterizerResources,
    binning_resources: &StrandBinningBuffers,
) -> Result<StrandBinningBindGroup, ()> {
    let strand_points_buffer = binning_resources.vertex_buffer.as_ref().ok_or(())?;
    let index_buffer = binning_resources.index_buffer.as_ref().ok_or(())?;
    let strand_metadata_buffer = binning_resources.meta_buffer.as_ref().ok_or(())?;

    let artifacts = binning_resources.artifacts.get(entity).ok_or(())?;

    let froxel_config_buffer = raster_resources
        .froxel_config_buffer
        .get(entity)
        .ok_or(())?;

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
                resource: artifacts.tile_counts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_CONFIG,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::VIEW_UNIFORM,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: layouts::binning::LIGHT_UNIFORM,
                resource: light_uniform.clone(),
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
                resource: artifacts.tile_counts_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::TILE_OFFSETS_BUFFER,
                resource: artifacts.tile_offsets_buffer.as_entire_binding(),
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
                resource: artifacts.tile_offsets_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::CURRENT_TILE_WRITE_INDICES,
                resource: artifacts
                    .current_tile_write_indices_buffer
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
                resource: artifacts
                    .current_tile_write_indices_buffer
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_TILE_BUFFER,
                resource: artifacts.packed_segments_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::FROXEL_CONFIG,
                resource: froxel_config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: layouts::binning::VIEW_UNIFORM,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: layouts::binning::LIGHT_UNIFORM,
                resource: light_uniform.clone(),
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
    return Ok(StrandBinningBindGroup {
        count_bind_group,
        scan_bind_group,
        init_placement_idx_bind_group,
        place_bind_group,
        offsets: vec![view_uniform_offset.offset, light_uniform_offset.offset],
    });
}

pub fn run_binning_pass(
    entity: &Entity,
    render_device: &RenderDevice,
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    bind_groups: &StrandBinningBindGroup,
    buffers: &StrandBinningBuffers,
    pipelines: &StrandBinningPipeline,
    froxel_config: &FroxelConfig,
    num_strands_or_segments: u32,
    is_light_entity: bool,
) {
    let Some(artifacts) = buffers.artifacts.get(entity) else {
        warn!("No artifacts for entity: {:?}", entity);
        return;
    };

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

    debug!(
        "max_compute_workgroups_per_dimension: {:?}",
        max_compute_workgroups_per_dimension
    );
    let threads_per_workgroup = NUMBER_OF_THREADS_PER_WORKGROUP;

    // --- Clear count buffer (important!) ---
    // Use encoder.clear_buffer(...) or a small compute shader pass
    encoder.clear_buffer(&artifacts.tile_counts_buffer, 0, None); // Clear whole buffer

    // --- Pass 1: Count ---
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Strand Count"),
            ..default()
        });
        let pipeline_id = if is_light_entity {
            pipelines.count_pipeline_shadows
        } else {
            pipelines.count_pipeline
        };
        let Some(count_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            warn!("Count pipeline not found");
            return;
        };
        pass.set_pipeline(count_pipeline);
        pass.set_bind_group(0, &bind_groups.count_bind_group, &bind_groups.offsets);
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
        debug!("Scan rounds: {:?}", rounds);

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
        // debug!("Scan last: load_base: {}, save_base: {}", load_base, save_base);
        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));
        // Set push constant for total count offset if separate buffer not used
        pass.set_push_constants(0, bytemuck::bytes_of(&total_count_buffer_offset));
        pass.dispatch_workgroups(1, 1, 1);

        // Scan Prefix (Propagate scan results back down)
        pass.set_pipeline(scan_prfx_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
            // debug!("Scan prefix: load_base: {}, save_base: {}", save_base, load_base);
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
        let pipeline_id = if is_light_entity {
            pipelines.place_pipeline_shadows
        } else {
            pipelines.place_pipeline
        };
        let Some(place_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            warn!("Place pipeline not found");
            return;
        };
        pass.set_pipeline(place_pipeline);
        pass.set_bind_group(0, &bind_groups.place_bind_group, &bind_groups.offsets);
        // Set push constants if needed
        // pass.set_push_constants(...);

        // Dispatch based on number of strands or segments (same as Pass 1)
        let workgroup_size_x = 64; // Match shader
        let num_workgroups_x = (num_strands_or_segments + workgroup_size_x - 1) / workgroup_size_x;
        pass.dispatch_workgroups(num_workgroups_x, 1, 1);
        // pass.dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
    }

    // --- Binning complete ---
    // packed_segments_buffer and tile_offsets_buffer are ready for the rasterizer
}

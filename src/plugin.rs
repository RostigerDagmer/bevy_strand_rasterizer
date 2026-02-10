use bevy::{
    core_pipeline::core_3d::graph::Core3d,
    math::bounding::Aabb3d,
    pbr::{
        ExtractedDirectionalLight, GlobalClusterableObjectMeta, LightMeta, ShadowSamplers,
        ViewClusterBindings, ViewLightsUniformOffset, ViewShadowBindings,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::ExtractComponentPlugin,
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel},
        render_resource::{
            BindGroupLayout, Buffer, BufferDescriptor, BufferUsages, CachedComputePipelineId,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewUniformOffset, ViewUniforms},
    },
    shader::ShaderDefVal,
};

use crate::{
    allocator::*,
    components::*,
    dson::DsonAsset,
    pipelines::{
        binning::*, composite::*, prepass::*, raster::*, shading::*, shadows::*,
        sim::StrandSimulatorResources, task_contract::QUEUE_HEADER_WORDS,
    },
    resources::*,
    shader_types::*,
};

pub const MAX_TEXTURE_EXTENT: u32 = 8192; // for shading (TODO: get this from device limits)
const BINNING_POOL_CHUNK_SIZE: u32 = 32;
const BINNING_POOL_NUM_HEADS: u32 = 64;
const BINNING_POOL_MIN_CHUNKS: u32 = 16_384;

use lazy_static::lazy_static;

lazy_static! {
    static ref BIND_MAP: HashMap<SlabKind, u32> = [
        (SlabKind::Vert, 0),
        (SlabKind::Index, 1),
        (SlabKind::StrandMaterial, 5),
        (SlabKind::StrandGeo, 6),
        (SlabKind::StrandMeta, 7),
    ]
    .iter()
    .copied()
    .collect();
}

lazy_static! {
    static ref LABEL_MAP: HashMap<SlabKind, &'static str> = [
        (SlabKind::Vert, "VERTICES"),
        (SlabKind::Index, "INDICES"),
        (SlabKind::StrandMaterial, "STRAND_MATERIALS"),
        (SlabKind::StrandGeo, "STRAND_GEOS"),
        (SlabKind::StrandMeta, "STRAND_METADATA"),
    ]
    .iter()
    .copied()
    .collect();
}

#[derive(Resource)]
pub struct AllocatorDebugPipeline {
    pipeline: Option<CachedComputePipelineId>,
    allocator_epoch: u64,
    shader: Handle<Shader>,
}

impl FromWorld for AllocatorDebugPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/allocator_debug_noop.wgsl");

        Self {
            pipeline: None,
            allocator_epoch: u64::MAX,
            shader,
        }
    }
}

fn prepare_allocator_debug_pipeline(
    mut debug_pipeline: ResMut<AllocatorDebugPipeline>,
    allocator: Res<GpuPagingAllocator>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(buffer_layout) = &allocator.buffer_bind_group_layout else {
        return;
    };
    let Some(table_layout) = &allocator.pagetable_bind_group_layout else {
        return;
    };
    if debug_pipeline.pipeline.is_some()
        && debug_pipeline.allocator_epoch == allocator.bindgroups_epoch
    {
        return;
    }

    let layout = vec![buffer_layout.clone(), table_layout.clone()];
    info!("layout: {:?}", layout);

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("allocator_debug_noop_pipeline".into()),
        layout,
        shader: debug_pipeline.shader.clone(),
        shader_defs: allocator.shader_defs(),
        push_constant_ranges: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });
    info!(
        "Debug pipeline queued for allocator epoch {}",
        allocator.bindgroups_epoch
    );

    debug_pipeline.pipeline = Some(pipeline);
    debug_pipeline.allocator_epoch = allocator.bindgroups_epoch;
}

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
            ExtractComponentPlugin::<StrandMaterial>::default(),
            GpuPagingAllocatorPlugin,
        ));
        app.init_resource::<StrandAssetResources>();
        app.add_systems(
            Update,
            (
                set_strand_geometry,
                flag_realloc_on_view_change,
                flag_realloc_on_config_change,
                flag_realloc_on_tie_change,
                tie_view_to_froxel_config,
            ),
        );
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(GpuPagingAllocatorSettings {
            label_map: LABEL_MAP.clone(),
            bind_map: BIND_MAP.clone(),
            buffer_group_idx: 0,
            table_group_idx: 1,
        });
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandRasterizerPipeline>();
        render_app.init_resource::<StrandShadingPipeline>();
        render_app.init_resource::<StrandShadingResources>();
        render_app.init_resource::<StrandShadowPipeline>();
        render_app.init_resource::<StrandShadowResources>();
        render_app.init_resource::<StrandBinningPipeline>();
        render_app.init_resource::<StrandBinningBuffers>();
        render_app.init_resource::<StrandPrepassResources>();
        render_app.init_resource::<ComputeInvocationDims>();
        render_app.init_resource::<StrandPrepassPipeline>();
        render_app.init_resource::<CompositionPipeline>();
        render_app.init_resource::<AllocatorDebugPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                use_deep_opacity_maps,
                use_prepass_buffers,
                // update_material_buffer,
                // use_strand_geometry.after(prepare_view_uniforms),
            )
                .chain()
                .in_set(RenderSystems::Prepare),),
        );
        // render_app.add_systems(
        //     Render,
        //     prepare_allocator_debug_pipeline.after(RenderSystems::PrepareBindGroups),
        // );
        render_app.add_systems(
            Render,
            update_strand_prepass_pipeline.after(RenderSystems::PrepareBindGroups),
        );
        render_app
            // .add_render_graph_node::<AllocatorDebugNode>(Core3d, AllocatorDebugLabel)
            .add_render_graph_node::<WorkPreparationNode>(Core3d, WorkPreparationLabel)
            // .add_render_graph_node::<StrandRasterizerNode>(Core3d, StrandRasterizerLabel)
            // .add_render_graph_node::<StrandShadingNode>(Core3d, StrandShadingLabel)
            // .add_render_graph_node::<StrandShadowRasterizerNode>(
            //     Core3d,
            //     StrandShadowRasterizerLabel,
            // )
            .add_render_graph_node::<CompositionNode>(Core3d, CompositionLabel)
            // connect nodes
            // .add_render_graph_edge(
            //     Core3d,
            //     bevy::core_pipeline::core_3d::graph::Node3d::EndPrepasses,
            //     StrandShadowRasterizerLabel,
            // )
            // .add_render_graph_edge(Core3d, StrandShadowRasterizerLabel, StrandShadingLabel)
            // .add_render_graph_edge(Core3d, StrandShadingLabel, StrandRasterizerLabel) // run after ALL shadow maps are present
            // .add_render_graph_edge(
            //     Core3d,
            //     StrandRasterizerLabel,
            //     CompositionLabel, // Run after strand rasterization
            // )
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::EndMainPass, // Run before composition
                WorkPreparationLabel,
            )
            // .add_render_graph_edge(Core3d, AllocatorDebugLabel, CompositionLabel)
            .add_render_graph_edge(Core3d, WorkPreparationLabel, CompositionLabel)
            .add_render_graph_edge(
                Core3d,
                CompositionLabel, // Run after composition
                bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing, // Before standard post-processing
            );
    }
}

fn use_prepass_buffers(
    geometry_query: Query<&StrandGeometry>,
    device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut prepass_resources: ResMut<StrandPrepassResources>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
) {
    let mut total_strands = 0u32;
    let mut geo_count = 0u32;
    for geom in &geometry_query {
        total_strands = total_strands.saturating_add(geom.strand_count);
        geo_count = geo_count.saturating_add(1);
    }

    if total_strands == 0 {
        return;
    }

    raster_resources.strand_count = Some(total_strands);

    let prepass_capacity = total_strands.next_power_of_two().max(1024);
    let binning_capacity = prepass_capacity;
    let geo_capacity = geo_count.next_power_of_two().max(1024);

    let needs_realloc = prepass_resources.prepass_queue.is_none()
        || prepass_resources.binning_queue.is_none()
        || prepass_resources.visibility_flags_buffer.is_none()
        || prepass_resources.visible_geos_buffer.is_none()
        || prepass_resources.geos_prefix_buffer.is_none()
        || prepass_resources.indirect_args.is_none()
        || prepass_resources.chunk_pool.is_none()
        || prepass_resources.free_heads.is_none()
        || prepass_resources.prepass_task_capacity < prepass_capacity
        || prepass_resources.binning_task_capacity < binning_capacity
        || prepass_resources.geo_capacity < geo_capacity;

    if needs_realloc {
        let prepass_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (prepass_capacity as u64) * (std::mem::size_of::<FinePrepassTask>() as u64);
        let binning_bytes = (QUEUE_HEADER_WORDS * std::mem::size_of::<u32>()) as u64
            + (binning_capacity as u64) * (std::mem::size_of::<BinningTask>() as u64);
        let geo_bytes = (geo_capacity as u64) * (std::mem::size_of::<u32>() as u64);
        let geo_prefix_bytes = ((geo_capacity as u64) + 1) * (std::mem::size_of::<u32>() as u64);
        let chunk_count = binning_capacity.max(BINNING_POOL_MIN_CHUNKS);
        let chunk_stride_bytes =
            (2u64 + BINNING_POOL_CHUNK_SIZE as u64) * std::mem::size_of::<u32>() as u64;
        let chunk_pool_bytes = chunk_count as u64 * chunk_stride_bytes;

        prepass_resources.prepass_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_queue"),
            size: prepass_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.binning_queue = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_queue"),
            size: binning_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.visibility_flags_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_visible_flags"),
            size: geo_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.visible_geos_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_visible_geos"),
            size: geo_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.geos_prefix_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_geo_prefix"),
            size: geo_prefix_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.indirect_args = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_prepass_indirect_args"),
            size: (3 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.chunk_pool = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_chunk_pool"),
            size: chunk_pool_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        prepass_resources.free_heads = Some(device.create_buffer(&BufferDescriptor {
            label: Some("strand_binning_free_heads"),
            size: (BINNING_POOL_NUM_HEADS as u64) * std::mem::size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        prepass_resources.prepass_task_capacity = prepass_capacity;
        prepass_resources.binning_task_capacity = binning_capacity;
        prepass_resources.geo_capacity = geo_capacity;

        info!(
            "Allocated prepass buffers: strands={} geos={} prepass_cap={} binning_cap={}",
            total_strands, geo_count, prepass_capacity, binning_capacity
        );
    }

    let zero_queue_hdr = [0u32, 0u32];
    let zero_dispatch = [0u32, 1u32, 1u32];

    if let Some(queue_buf) = &prepass_resources.prepass_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(queue_buf) = &prepass_resources.binning_queue {
        render_queue.write_buffer(queue_buf, 0, bytemuck::cast_slice(&zero_queue_hdr));
    }
    if let Some(indirect) = &prepass_resources.indirect_args {
        render_queue.write_buffer(indirect, 0, bytemuck::cast_slice(&zero_dispatch));
    }
    if let Some(free_heads) = &prepass_resources.free_heads {
        let zero_heads = vec![0u32; BINNING_POOL_NUM_HEADS as usize];
        render_queue.write_buffer(free_heads, 0, bytemuck::cast_slice(&zero_heads));
    }
}

#[derive(Debug, Clone, Default)]
pub struct WorkPreparationNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct WorkPreparationLabel;

impl Node for WorkPreparationNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let allocator = world.resource::<GpuPagingAllocator>();
        let prepass_pipeline = world.resource::<StrandPrepassPipeline>();
        let prepass_resources = world.resource::<StrandPrepassResources>();
        let invocation_dims = world.resource::<ComputeInvocationDims>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let light_meta = world.resource::<LightMeta>();
        let global_clusterable_object_meta = world.resource::<GlobalClusterableObjectMeta>();

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            return Ok(());
        };

        let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity)
        else {
            return Ok(());
        };

        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            return Ok(());
        };

        let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
            return Ok(());
        };

        let Some(clusterable_objects) = global_clusterable_object_meta
            .gpu_clusterable_objects
            .binding()
        else {
            return Ok(());
        };

        let Some(view_cluster_bindings) = world.get::<ViewClusterBindings>(view_entity) else {
            return Ok(());
        };

        let Some(cluster_indices_binding) =
            view_cluster_bindings.clusterable_object_index_lists_binding()
        else {
            return Ok(());
        };

        let Some(cluster_offsets_binding) = view_cluster_bindings.offsets_and_counts_binding()
        else {
            return Ok(());
        };

        let Ok((prepass_bind_group, dynamic_offsets)) = create_prepass_bind_group(
            render_device,
            prepass_pipeline,
            &prepass_resources,
            &view_binding,
            &light_binding,
            view_uniform_offset,
            view_light_uniform_offset,
            &cluster_indices_binding,
            &cluster_offsets_binding,
            &clusterable_objects,
        ) else {
            warn!("Failed to create prepass bind groups.");
            return Ok(());
        };
        let Some(indirect_args) = prepass_resources.indirect_args.as_ref() else {
            return Ok(());
        };
        info!("Running prepass");
        run_prepass(
            render_context,
            pipeline_cache,
            prepass_pipeline,
            allocator,
            invocation_dims,
            &prepass_bind_group,
            &dynamic_offsets,
            indirect_args,
        );

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct AllocatorDebugNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct AllocatorDebugLabel;

impl Node for AllocatorDebugNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let allocator = world.resource::<GpuPagingAllocator>();
        let debug_pipeline = world.resource::<AllocatorDebugPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(buffer_bind_group) = allocator.buffer_bind_group.as_ref() else {
            warn!("buffer bind group not present");
            return Ok(());
        };
        let Some(table_bind_group) = allocator.pagetable_bind_group.as_ref() else {
            warn!("table bind group not present");
            return Ok(());
        };
        let Some(pipeline_id) = debug_pipeline.pipeline else {
            warn!("debug pipeline id not present");
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            warn!("debug pipeline not present");
            return Ok(());
        };
        info!("Allocator: {:?}", allocator);
        // info!("Buffer bind group: {:?}", buffer_bind_group);
        // info!("Table bind group: {:?}", table_bind_group);
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Allocator Debug Noop"),
                    ..default()
                });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(allocator.buffer_group_idx, buffer_bind_group, &[]);
        pass.set_bind_group(allocator.table_group_idx, table_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        info!("Debug dispatch");
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrandShadowRasterizerNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandShadowRasterizerLabel;

impl Node for StrandShadowRasterizerNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world.contains_resource::<StrandRasterizerResources>() {
            return Ok(());
        }
        let view_entity = graph.view_entity(); // Get the entity this node instance is running for
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let shadow_pipeline = world.resource::<StrandShadowPipeline>();
        let binning_pipeline = world.resource::<StrandBinningPipeline>();
        let shadow_resources = world.resource::<StrandShadowResources>();
        let shading_resources = world.resource::<StrandShadingResources>();
        let binning_buffers = world.resource::<StrandBinningBuffers>();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms
        let light_meta = world.resource::<LightMeta>(); // Get light meta
        let global_clusterable_object_meta = world.resource::<GlobalClusterableObjectMeta>();
        let shadow_samplers = world.resource::<ShadowSamplers>();

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewUniformOffset",
                view_entity
            );
            return Ok(());
        };

        let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity)
        else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewLightUniformOffset",
                view_entity
            );
            return Ok(());
        };

        // --- Check Prerequisites ---
        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            warn!("ViewUniforms binding not available.");
            return Ok(());
        };

        let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without LightBinding",
                view_entity
            );
            return Ok(());
        };

        let Some(clusterable_objects) = global_clusterable_object_meta
            .gpu_clusterable_objects
            .binding()
        else {
            warn!("GlobalClusterableObjectMeta binding not available.");
            return Ok(());
        };

        let Some(view_cluster_bindings) = world.get::<ViewClusterBindings>(view_entity) else {
            warn!(
                "Node running on view {:?} without ViewClusterBindings",
                view_entity
            );
            // This might be expected if clustering isn't enabled/used for this view?
            return Ok(()); // Adjust handling if necessary
        };

        let Some(view_shadow_bindings) = world.get::<ViewShadowBindings>(view_entity) else {
            warn!(
                "Node running on view {:?} without ViewShadowBindings",
                view_entity
            );
            // This is expected for views rendering shadow maps, but required for views sampling them.
            // If your node ONLY samples shadows, this might be an error.
            // If your node might run on shadow views, handle appropriately.
            return Ok(()); // Adjust handling if necessary
        };
        let Some(cluster_indices_binding) =
            view_cluster_bindings.clusterable_object_index_lists_binding()
        else {
            warn!(
                "ViewClusterBindings clusterable_object_index_lists_binding not available for view {:?}",
                view_entity
            );
            return Ok(());
        };

        let Some(cluster_offsets_binding) = view_cluster_bindings.offsets_and_counts_binding()
        else {
            warn!(
                "ViewClusterBindings offsets_and_counts_binding not available for view {:?}",
                view_entity
            );
            return Ok(());
        };

        for entity in raster_resources.froxel_config_buffer.keys() {
            debug!("Dispatching for entity: {:?}", entity);
            // Get the dimensions to calculate dispatch size
            let Some(frustrum) = raster_resources.frustrum_config.get(entity) else {
                warn!("No frustum size defined.");
                return Ok(());
            };

            // Get the strand count for dispatch dimensions
            let Some(strand_count) = raster_resources.strand_count else {
                warn!("No strand count set.");
                return Ok(());
            };

            if *entity != view_entity {
                debug!("light entity: {:?}", entity);
                let Ok((shadows_bindgroup, dynamic_offsets)) = &create_strand_shadow_bind_group(
                    entity,
                    render_device,
                    shadow_pipeline,
                    shadow_resources,
                    shading_resources,
                    &raster_resources,
                    binning_buffers,
                    &view_binding,
                    &light_binding,
                    view_uniform_offset,
                    view_light_uniform_offset,
                    &cluster_indices_binding,
                    &cluster_offsets_binding,
                    &clusterable_objects,
                    shadow_samplers,
                    (
                        &view_shadow_bindings.point_light_depth_texture_view,
                        &view_shadow_bindings.directional_light_depth_texture_view,
                    ),
                ) else {
                    warn!("Failed to create strand shadow bind group.");
                    continue;
                };
                // Binning bind group (shadow pass)
                let Ok(strand_binning_bind_groups) = &create_strand_binning_bind_group(
                    &entity,
                    render_device,
                    binning_pipeline,
                    view_binding.clone(),
                    view_uniform_offset,
                    light_binding.clone(),
                    view_light_uniform_offset,
                    raster_resources,
                    binning_buffers,
                ) else {
                    warn!("Failed to create strand binning bind group for shadows .");
                    return Ok(());
                };

                run_binning_pass(
                    &entity,
                    render_device,
                    render_context,
                    pipeline_cache,
                    strand_binning_bind_groups,
                    binning_buffers,
                    binning_pipeline,
                    &frustrum,
                    strand_count,
                    true,
                );

                run_shadow_pass(
                    render_context,
                    pipeline_cache,
                    shadow_pipeline,
                    &frustrum,
                    &raster_resources,
                    &shadows_bindgroup,
                    &dynamic_offsets,
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct DomPunchThroughNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct DomPunchThroughLabel;

impl Node for DomPunchThroughNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // pass
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrandShadingNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandShadingLabel;

impl Node for StrandShadingNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Check if we have resources
        if !world.contains_resource::<StrandRasterizerResources>() {
            return Ok(());
        }
        let view_entity = graph.view_entity(); // Get the entity this node instance is running for

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let shading_pipeline = world.resource::<StrandShadingPipeline>();
        let binning_buffers = world.resource::<StrandBinningBuffers>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let shading_resources = world.resource::<StrandShadingResources>();
        let shadow_resources = world.resource::<StrandShadowResources>();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms
        let light_meta = world.resource::<LightMeta>(); // Get light meta
        let global_clusterable_object_meta = world.resource::<GlobalClusterableObjectMeta>();
        let shadow_samplers = world.resource::<ShadowSamplers>();

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewUniformOffset",
                view_entity
            );
            return Ok(());
        };

        let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity)
        else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewLightUniformOffset",
                view_entity
            );
            return Ok(());
        };

        // --- Check Prerequisites ---
        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            warn!("ViewUniforms binding not available.");
            return Ok(());
        };

        let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without LightBinding",
                view_entity
            );
            return Ok(());
        };
        let Some(clusterable_objects) = global_clusterable_object_meta
            .gpu_clusterable_objects
            .binding()
        else {
            warn!("GlobalClusterableObjectMeta binding not available.");
            return Ok(());
        };

        let Some(view_cluster_bindings) = world.get::<ViewClusterBindings>(view_entity) else {
            warn!(
                "Node running on view {:?} without ViewClusterBindings",
                view_entity
            );
            // This might be expected if clustering isn't enabled/used for this view?
            return Ok(()); // Adjust handling if necessary
        };

        let Some(cluster_indices_binding) =
            view_cluster_bindings.clusterable_object_index_lists_binding()
        else {
            warn!(
                "ViewClusterBindings clusterable_object_index_lists_binding not available for view {:?}",
                view_entity
            );
            return Ok(());
        };

        let Some(cluster_offsets_binding) = view_cluster_bindings.offsets_and_counts_binding()
        else {
            warn!(
                "ViewClusterBindings offsets_and_counts_binding not available for view {:?}",
                view_entity
            );
            return Ok(());
        };

        // Shading bind group
        let Ok((shading_bind_group, shading_group_offsets)) = create_strand_shading_bind_group(
            &view_entity,
            render_device,
            &shading_pipeline,
            raster_resources,
            &shading_resources,
            &shadow_resources,
            &binning_buffers,
            &view_binding,
            &light_binding,
            view_uniform_offset,
            view_light_uniform_offset,
            &cluster_indices_binding,
            &cluster_offsets_binding,
            &clusterable_objects,
            shadow_samplers,
        ) else {
            warn!("Failed to create strand shading bind group.");
            return Ok(());
        };

        run_shading_pass(
            render_context,
            pipeline_cache,
            shading_pipeline,
            shading_resources,
            &shading_bind_group,
            &shading_group_offsets,
        );

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrandRasterizerNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandRasterizerLabel;

impl Node for StrandRasterizerNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Check if we have resources
        if !world.contains_resource::<StrandRasterizerResources>() {
            return Ok(());
        }
        let view_entity = graph.view_entity(); // Get the entity this node instance is running for

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let binning_pipeline = world.resource::<StrandBinningPipeline>();
        // let shading_pipeline = world.resource::<StrandShadingPipeline>();
        // let shadow_pipeline = world.resource::<StrandShadowPipeline>();
        let raster_pipeline = world.resource::<StrandRasterizerPipeline>();
        let binning_buffers = world.resource::<StrandBinningBuffers>();
        let shading_resources = world.resource::<StrandShadingResources>();
        let shadow_resources = world.resource::<StrandShadowResources>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms
        let light_meta = world.resource::<LightMeta>(); // Get light meta
        // let global_clusterable_object_meta = world.resource::<GlobalClusterableObjectMeta>();
        // let shadow_samplers = world.resource::<ShadowSamplers>();

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewUniformOffset",
                view_entity
            );
            return Ok(());
        };

        let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity)
        else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewLightUniformOffset",
                view_entity
            );
            return Ok(());
        };

        // --- Check Prerequisites ---
        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            warn!("ViewUniforms binding not available.");
            return Ok(());
        };

        let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without LightBinding",
                view_entity
            );
            return Ok(());
        };

        // Get the dimensions to calculate dispatch size
        let Some(frustrum) = raster_resources.frustrum_config.get(&view_entity) else {
            warn!("No frustum size defined.");
            return Ok(());
        };

        // Get the strand count for dispatch dimensions
        let Some(strand_count) = raster_resources.strand_count else {
            warn!("No strand count set.");
            return Ok(());
        };

        // Binning bind group
        let Ok(strand_binning_bind_groups) = &create_strand_binning_bind_group(
            &view_entity,
            render_device,
            binning_pipeline,
            view_binding.clone(),
            view_uniform_offset,
            light_binding.clone(),
            view_light_uniform_offset,
            raster_resources,
            binning_buffers,
        ) else {
            warn!("Failed to create strand binning bind group.");
            return Ok(());
        };

        // Raster bind group
        let Ok((raster_bind_group, raster_group_offsets)) = create_strand_raster_bind_group(
            &view_entity,
            render_device,
            &raster_pipeline,
            &raster_resources,
            &binning_buffers,
            &shading_resources,
            &shadow_resources,
            &view_binding,
            &light_binding,
            &view_uniform_offset,
            &view_light_uniform_offset,
        ) else {
            warn!("Failed to create strand raster bind group.");
            return Ok(());
        };

        run_binning_pass(
            &view_entity,
            render_device,
            render_context,
            pipeline_cache,
            strand_binning_bind_groups,
            binning_buffers,
            binning_pipeline,
            &frustrum,
            strand_count,
            false,
        );

        run_raster_pass(
            render_context,
            pipeline_cache,
            raster_pipeline,
            &frustrum,
            raster_resources,
            shading_resources,
            &raster_bind_group,
            &raster_group_offsets,
        );

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrandSimulationNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct StrandSimulationLabel;

impl Node for StrandSimulationNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Check if we have resources
        if !world.contains_resource::<StrandSimulatorResources>() {
            return Ok(());
        }
        let view_entity = graph.view_entity(); // Get the entity this node instance is running for

        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        Ok(())
    }
}

// main world buffer initialization
// packs StrandGeometry
fn set_strand_geometry(
    query: Query<(Entity, &StrandAsset, &StrandMaterial), Without<StrandGeometry>>,
    assets: Res<Assets<DsonAsset>>,
    mut storage_buffers: ResMut<Assets<VirtualShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_asset, material) in query.iter() {
        let Some(asset) = assets.get(&strand_asset.handle) else {
            continue;
        };

        let Some(geometry_library) = &asset.dson_file.geometry_library else {
            warn!("Geometry library not found for entity: {:?}", entity);
            continue;
        };

        if geometry_library.is_empty() {
            warn!("Geometry library is empty for entity: {:?}", entity);
            continue;
        }

        let geometry = &geometry_library[0];
        // Extract vertices
        // TODO: time this. Could also be done in a compute shader
        let mut aabb = Aabb3d::from_point_cloud(
            Isometry3d::IDENTITY,
            geometry.vertices.values.iter().cloned(),
        ); // todo: pass transform
        aabb.min *= 0.0254; // TODO: pass transform to shaders
        aabb.max *= 0.0254; // TODO: pass transform to shaders
        let vertices: Vec<[f32; 4]> = geometry
            .vertices
            .values
            .clone()
            .iter()
            .map(|v| {
                [
                    v[0] * 0.0254, // TODO: pass transform to shaders
                    v[1] * 0.0254, // TODO: pass transform to shaders
                    v[2] * 0.0254, // TODO: pass transform to shaders
                    1.0,
                ]
            })
            .collect();

        // Extract indices from polyline_list
        let Some(polyline_list) = &geometry.polyline_list else {
            warn!("Polyline list not found for entity: {:?}", entity);
            continue;
        };

        // Flatten the polyline indices
        // For each strand in values, skip first two elements (group_idx, mat_group_idx)
        // and collect the vertex indices
        let packed_strand_info =
            polyline_list
                .values
                .iter()
                .fold((Vec::new(), Vec::new(), 0), |mut acc, strand| {
                    let strand_indices = &strand[2..];
                    acc.0.extend_from_slice(strand_indices);
                    if strand_indices.len() > acc.2 {
                        acc.2 = strand_indices.len();
                    }
                    match acc.1.last().copied() {
                        Some((last_strand_count, last_strand_offset)) => {
                            acc.1.push((
                                strand_indices.len() as u32,
                                last_strand_offset + last_strand_count,
                            ));
                        }
                        None => acc.1.push((strand_indices.len() as u32, 0)),
                    }
                    acc
                });

        let indices = packed_strand_info.0;
        let meta: Vec<StrandMeta> = packed_strand_info
            .1
            .into_iter()
            .map(StrandMeta::from)
            .collect();
        let max_segments_in_strand = packed_strand_info.2 as u32;

        let geos_data = vec![StrandGeo::new(
            polyline_list.values.len() as u32,
            max_segments_in_strand,
            aabb,
        )];

        let vertex_buffer = VirtualShaderStorageBuffer::from((SlabKind::Vert, vertices));
        let index_buffer = VirtualShaderStorageBuffer::from((SlabKind::Index, indices));
        let meta_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandMeta, meta));
        let geo_buffer = VirtualShaderStorageBuffer::from((SlabKind::StrandGeo, geos_data));
        let material_buffer =
            VirtualShaderStorageBuffer::from((SlabKind::StrandMaterial, vec![material]));

        let vertex_buffer_handle = storage_buffers.add(vertex_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);
        let meta_buffer_handle = storage_buffers.add(meta_buffer);
        let geo_buffer_handle = storage_buffers.add(geo_buffer);
        let material_buffer_handle = storage_buffers.add(material_buffer);

        info!("set strand geometry: {:?}", vertex_buffer_handle);

        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle,
            meta: meta_buffer_handle,
            geos: geo_buffer_handle,
            materials: material_buffer_handle,
            strand_count: polyline_list.values.len() as u32,
            max_segments_in_strand,
            aabb,
        });
    }
}

// uploads changed material paramters to the buffer
// TODO
// pub fn update_material_buffer(
//     query: Query<(Entity, &StrandGeometry, &StrandMaterial)>,
//     storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
//     render_queue: Res<RenderQueue>,
// ) {
//     for (entity, geometry, material) in query.iter() {
//         let Some(material_buffer) = storage_buffers.get(&geometry.materials) else {
//             warn!("Material storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         let material_bytes = bytemuck::bytes_of(material);
//         render_queue.write_buffer(
//             &material_buffer.buffer, // Get the underlying wgpu::Buffer
//             0,                       // Offset in the buffer to start writing (0 for the start)
//             material_bytes,          // The byte slice to write
//         );
//     }
// }

// Create froxel configuration uniform buffer
pub fn create_froxel_config_buffer(device: &RenderDevice, config: &FroxelConfig) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("strand_froxel_config_buffer"),
        size: std::mem::size_of::<FroxelConfig>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // Initialize with configuration
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    mapped.copy_from_slice(bytemuck::bytes_of(config));
    drop(mapped);
    buffer.unmap();
    buffer
}

fn flag_realloc_on_view_change(
    mut commands: Commands,
    cams: Query<(Entity, Option<&TieFroxelsToView>), Changed<ExtractedView>>,
    // or listen to WindowResized and map to camera(s)
) {
    for (e, _) in &cams {
        commands.entity(e).insert(NeedsRealloc);
    }
}

fn flag_realloc_on_config_change(mut commands: Commands, q: Query<Entity, Changed<FroxelConfig>>) {
    for e in &q {
        commands.entity(e).insert(NeedsRealloc);
    }
}

fn flag_realloc_on_tie_change(mut commands: Commands, q: Query<Entity, Changed<TieFroxelsToView>>) {
    for e in &q {
        commands.entity(e).insert(NeedsRealloc);
    }
}

fn tie_view_to_froxel_config(
    cams: Query<
        (Entity, &TieFroxelsToView, &ExtractedView, &mut FroxelConfig),
        Changed<ExtractedView>,
    >,
) {
    for (entity, tie_mode, extracted_view, mut config) in cams {
        info!("Viewport change: {:?}", extracted_view.viewport);
        let viewport_x = extracted_view.viewport.z;
        let viewport_y = extracted_view.viewport.w;
        match tie_mode {
            TieFroxelsToView::Fixed(v) => {}
            TieFroxelsToView::Native => {
                config.screen_width = viewport_x;
                config.screen_height = viewport_y;
            }
            TieFroxelsToView::Scaled(s) => {
                let w = ((viewport_x as f32) * s) as u32;
                let h = ((viewport_y as f32) * s) as u32;
                config.screen_width = w;
                config.screen_height = h;
            }
        }
    }
}

// render world buffer retrieval
fn use_froxel_buffer(
    query: Query<(Entity, &FroxelConfig), Or<(Added<FroxelConfig>, With<NeedsRealloc>)>>,
    device: Res<RenderDevice>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut binning_resources: ResMut<StrandBinningBuffers>,
) {
    for (entity, config) in query.iter() {
        let config_buffer = create_froxel_config_buffer(&device, config);
        let (
            tile_counts_buffer,
            tile_offsets_buffer,
            current_tile_write_indices_buffer,
            packed_segments_buffer,
        ) = prepare_binning_buffers(&device, config);

        let (texture, target_view) = recreate_render_target_texture(&device, config);
        let (texture, depth_view) = recreate_render_target_depth_texture(&device, config);

        info!("Recreated render target: {:?}", config);

        raster_resources.output_texture = Some(target_view);
        raster_resources.output_depth = Some(depth_view);
        // modify the resource
        raster_resources
            .froxel_buffer
            .insert(entity, packed_segments_buffer.clone());
        raster_resources
            .froxel_config_buffer
            .insert(entity, config_buffer);
        raster_resources
            .frustrum_config
            .insert(entity, config.clone());

        let artifact_buffers = StrandBinningArtifactBuffers {
            tile_counts_buffer,
            tile_offsets_buffer,
            current_tile_write_indices_buffer,
            packed_segments_buffer,
        };

        binning_resources.artifacts.insert(entity, artifact_buffers);

        debug!("Added froxel buffers to resource");
    }
}

fn use_deep_opacity_maps(
    query: Query<(Entity, &FroxelConfig), With<ExtractedDirectionalLight>>,
    device: Res<RenderDevice>,
    mut shadow_resources: ResMut<StrandShadowResources>,
) {
    for (entity, config) in query.iter() {
        if shadow_resources.dom_targets.contains_key(&entity) {
            continue;
        }
        let (
            (opacity_texture, opacity_view, opacity_sampler),
            (depth_texture, depth_view, depth_sampler),
        ) = create_strand_shadow_textures(
            &device,
            config.screen_width,
            config.screen_height,
            config.depth_slices,
        );
        shadow_resources
            .dom_targets
            .insert(entity, (opacity_view, depth_view));
        shadow_resources
            .dom_samplers
            .insert(entity, (opacity_sampler, depth_sampler));
        info!("Added deep opacity maps to resource");
    }
}

// should be done entirely in the extract schedule of GpuPaging
// render world buffer retrieval (deprecate)
// fn use_strand_geometry(
//     query: Query<(Entity, &StrandGeometry)>,
//     storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
//     device: Res<RenderDevice>,
//     asset_resources: Res<StrandAssetResources>,
//     mut raster_resources: ResMut<StrandRasterizerResources>,
//     mut binning_resources: ResMut<StrandBinningBuffers>,
//     mut shading_resources: ResMut<StrandShadingResources>,
//     mut prepass_resources: ResMut<StrandPrepassResources>
// ) {
//     // This is an example of how to retrieve the shader storage buffer created in the main world above
//     // and use it in the render world.
//     for (entity, geometry) in query.iter() {
//         // if raster_resources.bind_group.is_some() {
//         //     continue;
//         // }

//         // --- Raster resources ---
//         debug!("Using strand geometry for entity: {:?}", entity);
//         let Some(index_storage_buffer) = storage_buffers.get(&geometry.indices) else {
//             warn!("Index storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         debug!("[{:?}] Index storage buffer found.", entity);
//         let Some(vertex_storage_buffer) = storage_buffers.get(&geometry.vertices) else {
//             warn!("Vertex storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         let Some(meta_storage_buffer) = storage_buffers.get(&geometry.meta) else {
//             warn!("Meta storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         let Some(geo_storage_buffer) = storage_buffers.get(&geometry.geos) else {
//             warn!("Geo storage buffer not found for entity: {:?}", entity);
//             continue;
//         };
//         let Some(material_buffer) = storage_buffers.get(&geometry.materials) else {
//             warn!("Material storage buffer not found for entity: {:?}", entity);
//             continue;
//         };

//         debug!(
//             "[{:?}] Vertex storage buffer found: {:?}",
//             entity, vertex_storage_buffer.buffer
//         );

//         binning_resources.vertex_buffer = Some(vertex_storage_buffer.buffer.clone());
//         binning_resources.meta_buffer = Some(meta_storage_buffer.buffer.clone());
//         binning_resources.index_buffer = Some(index_storage_buffer.buffer.clone());
//         binning_resources.geos_buffer = Some(geo_storage_buffer.buffer.clone());
//         // Prepass (-> to replace the binning bindings)
//         prepass_resources.vertex_buffer = Some(vertex_storage_buffer.buffer.clone());
//         prepass_resources.meta_buffer = Some(meta_storage_buffer.buffer.clone());
//         prepass_resources.index_buffer = Some(index_storage_buffer.buffer.clone());
//         prepass_resources.geos_buffer = Some(geo_storage_buffer.buffer.clone());

//         raster_resources.strand_count = Some(geometry.strand_count);

//         let (shading_buffer, shading_buffer_view) = create_shading_target_texture(
//             &device,
//             geometry.strand_count,
//             geometry.max_segments_in_strand,
//         );
//         shading_resources.output_texture = Some(shading_buffer_view);
//         shading_resources.strand_count = Some(geometry.strand_count);
//         shading_resources.materials = Some(material_buffer.buffer.clone());
//         shading_resources.max_segments_in_strand = Some(geometry.max_segments_in_strand);

//         debug!("Created bind group for strand rasterizer");
//     }

//     // pool buffer assignment
//     if let Some(prepass_queue) = &asset_resources.pool.prepass_queue {
//         prepass_resources.prepass_queue = Some(prepass_queue.clone());
//     } else {
//         warn!("Prepass queue in asset_resources not ready.");
//     }
//     if let Some(prepass_queue) = &asset_resources.pool.prepass_queue {
//         prepass_resources.prepass_queue = Some(prepass_queue.clone());
//     } else {
//         warn!("Prepass queue in asset_resources not ready.");
//     }
// }

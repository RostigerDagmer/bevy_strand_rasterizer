use bevy::{core_pipeline::core_3d::graph::Core3d, math::bounding::Aabb3d, pbr::{LightMeta, ViewLightsUniformOffset}, prelude::*, render::{extract_component::ExtractComponentPlugin, render_asset::RenderAssets, render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel}, render_resource::{BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferUsages, PipelineCache, ShaderType}, renderer::{RenderContext, RenderDevice}, storage::{GpuShaderStorageBuffer, ShaderStorageBuffer}, view::{prepare_view_uniforms, ViewUniform, ViewUniformOffset, ViewUniforms}, Render, RenderApp, RenderSet}};

use crate::{components::*, dson::DsonAsset, pipelines::{binning::*, composite::*, raster::*, shading::*}, resources::*, shader_types::*};

const MAX_NUMBER_OF_STRANDS: u32 = 1024 * 1024; // for physics
pub const MAX_TEXTURE_EXTENT: u32 = 8192; // for shading (TODO: get this from device limits)

pub struct StrandRasterizerPlugin;

impl Plugin for StrandRasterizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            // ExtractComponentPlugin::<Strands>::default(),
            ExtractComponentPlugin::<FroxelConfig>::default(),
            ExtractComponentPlugin::<StrandGeometry>::default(),
        ));
        app.init_resource::<StrandAssetResources>();
        app.add_systems(Update, set_strand_geometry);
    }
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // render_app.add_plugins(GetSubgroupSizePlugin);
        render_app.init_resource::<StrandRasterizerResources>();
        render_app.init_resource::<StrandRasterizerPipeline>();
        render_app.init_resource::<StrandShadingPipeline>();
        render_app.init_resource::<StrandShadingResources>();
        render_app.init_resource::<StrandBinningPipeline>();
        render_app.init_resource::<StrandBinningBuffers>();
        render_app.init_resource::<CompositionPipeline>();
        render_app.add_systems(
            Render,
            ((
                use_froxel_buffer,
                use_strand_geometry.after(prepare_view_uniforms),
            )
                .chain()
                .in_set(RenderSet::Prepare),),
        );
        render_app
            // Bevy's renderer uses a render graph which is a collection of nodes in a directed acyclic graph.
            // It currently runs on each view/camera and executes each node in the specified order.
            // It will make sure that any node that needs a dependency from another node
            // only runs when that dependency is done.
            //
            // Each node can execute arbitrary work, but it generally runs at least one render pass.
            // A node only has access to the render world, so if you need data from the main world
            // you need to extract it manually or with the plugin like above.
            // Add a [`Node`] to the [`RenderGraph`]
            // The Node needs to impl FromWorld (dealt with by derive(Default))
            // .add_render_graph_node::<SpatialHashingNode>(Core3d, SpatialHashingLabel)
            // .add_render_graph_node::<SimpleGpuSortNode>(Core3d, SimpleGpuSortNodeLabel)
            .add_render_graph_node::<StrandRasterizerNode>(Core3d, StrandRasterizerLabel)
            .add_render_graph_node::<CompositionNode>(Core3d, CompositionLabel)
            // .add_render_graph_edge(
            //     Core3d,
            //     SpatialHashingLabel,
            //     bevy::core_pipeline::core_3d::graph::Node3d::EndPrepasses,
            // )
            // .add_render_graph_edge(Core3d, SimpleGpuSortNodeLabel, SpatialHashingLabel)
            // .add_render_graph_edge(Core3d, StrandRasterizerLabel, SimpleGpuSortNodeLabel)
            .add_render_graph_edge(
                Core3d,
                bevy::core_pipeline::core_3d::graph::Node3d::EndMainPass,
                StrandRasterizerLabel,
            )
            .add_render_graph_edge(
                Core3d,
                StrandRasterizerLabel, // Run after strand rasterization
                CompositionLabel,
            )
            .add_render_graph_edge(
                Core3d,
                CompositionLabel, // Run after composition
                bevy::core_pipeline::core_3d::graph::Node3d::PostProcessing, // Before standard post-processing
            );
    }
}

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
        let shading_pipeline = world.resource::<StrandShadingPipeline>();
        let raster_pipeline = world.resource::<StrandRasterizerPipeline>();
        let binning_buffers = world.resource::<StrandBinningBuffers>(); // Get the buffers
        let shading_resources = world.resource::<StrandShadingResources>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms
        let light_meta = world.resource::<LightMeta>(); // Get light meta

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

        let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without LightBinding",
                view_entity
            );
            return Ok(());
        };

        // --- Check Prerequisites ---
        let Some(view_binding) = view_uniforms.uniforms.buffer() else {
            warn!("ViewUniforms binding not available.");
            return Ok(());
        };

        let view_binding = BindingResource::Buffer(BufferBinding {
            buffer: view_binding,
            offset: view_uniform_offset.offset as u64,
            size: Some(ViewUniform::min_size()),
        });
        // pass.set_bind_group(I, &mesh_view_bind_group.value, &offsets); // Reference

        let Some(vertex_buffer) = binning_buffers.vertex_buffer.as_ref() else {
            warn!("Vertex buffer handle not found in resources.");
            return Ok(());
        };
        let Some(index_buffer) = binning_buffers.index_buffer.as_ref() else {
            warn!("Index buffer handle not found in resources.");
            return Ok(());
        };
        let Some(meta_buffer) = binning_buffers.meta_buffer.as_ref() else {
            warn!("Meta buffer handle not found in resources.");
            return Ok(());
        };

        let Some(geos_buffer) = binning_buffers.geos_buffer.as_ref() else {
            warn!("Geos buffer handle not found in resources.");
            return Ok(());
        };

        let Some(tile_counts_buffer) = binning_buffers.tile_counts_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(tile_offsets_buffer) = binning_buffers.tile_offsets_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(current_tile_write_indices_buffer) =
            binning_buffers.current_tile_write_indices_buffer.as_ref()
        else {
            /* ... */
            return Ok(());
        };
        let Some(packed_segments_buffer) = binning_buffers.packed_segments_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(froxel_config_buffer) = raster_resources.froxel_config_buffer.as_ref() else {
            /* ... */
            return Ok(());
        };
        let Some(frustrum) = raster_resources.frustrum_config else {
            /* ... */
            return Ok(());
        };
        let Some(strand_count) = raster_resources.strand_count else {
            /* ... */
            return Ok(());
        };

        // Get the dimensions to calculate dispatch size
        let Some(frustrum) = raster_resources.frustrum_config else {
            warn!("No frustum size defined.");
            return Ok(());
        };

        let Some(strand_count) = raster_resources.strand_count else {
            warn!("No strand count set.");
            return Ok(());
        };

        let strand_binning_bind_groups = &create_strand_binning_bind_group(
            render_device,
            binning_pipeline,
            vertex_buffer,
            index_buffer,
            meta_buffer,
            view_binding.clone(),
            raster_resources,
            binning_buffers,
        );

        run_binning_pass(
            render_device,
            pipeline_cache,
            strand_binning_bind_groups,
            binning_buffers,
            binning_pipeline,
            render_context,
            &frustrum,
            strand_count,
        );

        // Shading pass
        if let Some(output_texture) = &shading_resources.output_texture {
            let (shading_bind_group, shading_group_offsets) = create_strand_shading_bind_group(
                render_device,
                &shading_pipeline.bind_group_layout,
                vertex_buffer,
                index_buffer,
                meta_buffer,
                &output_texture,
                view_binding.clone(),
                light_binding,
                view_light_uniform_offset,
            );
            let raster_bind_group = create_strand_raster_bind_group(
                render_device,
                &raster_pipeline.bind_group_layout,
                vertex_buffer,
                index_buffer,
                tile_offsets_buffer,
                tile_counts_buffer,
                meta_buffer,
                packed_segments_buffer,
                raster_resources.output_texture.as_ref().unwrap(),
                froxel_config_buffer,
                view_binding.clone(),
                output_texture,
            );

            run_shading_pass(
                render_context,
                pipeline_cache,
                shading_pipeline,
                shading_resources,
                &shading_bind_group,
                &shading_group_offsets,
            );
            // Raster pass
            run_raster_pass(
                render_context,
                pipeline_cache,
                raster_pipeline,
                &frustrum,
                raster_resources,
                shading_resources,
                &raster_bind_group,
            );
        }

        Ok(())
    }
}




// main world buffer initialization
// packs StrandGeometry
fn set_strand_geometry(
    query: Query<(Entity, &StrandAsset), Without<StrandGeometry>>,
    assets: Res<Assets<DsonAsset>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut commands: Commands,
) {
    for (entity, strand_asset) in query.iter() {
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
        let aabb = Aabb3d::from_point_cloud(
            Isometry3d::IDENTITY,
            geometry.vertices.values.iter().cloned(),
        ); // todo: pass transform
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
        let vertex_buffer = ShaderStorageBuffer::from(vertices);
        // info!("Vertex buffer: {:?}", vertex_buffer);
        let vertex_buffer_handle = storage_buffers.add(vertex_buffer);

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

        let strand_indices = packed_strand_info.0;
        let strand_meta: Vec<StrandMeta> = packed_strand_info
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

        let index_buffer = ShaderStorageBuffer::from(strand_indices);
        let meta_buffer = ShaderStorageBuffer::from(strand_meta);
        let geo_buffer = ShaderStorageBuffer::from(geos_data);
        // info!("Index buffer: {:?}", index_buffer);
        let index_buffer_handle = storage_buffers.add(index_buffer);
        let meta_buffer_handle = storage_buffers.add(meta_buffer);
        let geo_buffer_handle = storage_buffers.add(geo_buffer);

        commands.entity(entity).insert(StrandGeometry {
            vertices: vertex_buffer_handle,
            indices: index_buffer_handle,
            meta: meta_buffer_handle,
            geos: geo_buffer_handle,
            strand_count: polyline_list.values.len() as u32,
            max_segments_in_strand,
            aabb,
        });
    }
}

// render world buffe retrieval
fn use_froxel_buffer(
    query: Query<(Entity, &FroxelConfig), Added<FroxelConfig>>,
    device: Res<RenderDevice>,
    pipeline: Res<StrandBinningPipeline>,
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

        let (texture, view) = create_render_target_texture(&device, config);
        raster_resources.output_texture = Some(view);
        // modify the resource
        raster_resources.froxel_buffer = Some(packed_segments_buffer.clone());
        raster_resources.froxel_config_buffer = Some(config_buffer);
        raster_resources.frustrum_config = Some(config.clone());

        binning_resources.tile_counts_buffer = Some(tile_counts_buffer);
        binning_resources.tile_offsets_buffer = Some(tile_offsets_buffer);
        binning_resources.current_tile_write_indices_buffer =
            Some(current_tile_write_indices_buffer);
        binning_resources.packed_segments_buffer = Some(packed_segments_buffer);

        info!("Added froxel buffers to resource");
    }
}

// render world buffer retrieval
fn use_strand_geometry(
    query: Query<(Entity, &StrandGeometry)>,
    storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    device: Res<RenderDevice>,
    raster_pipeline: Res<StrandRasterizerPipeline>,
    binning_pipeline: Res<StrandBinningPipeline>,
    mut raster_resources: ResMut<StrandRasterizerResources>,
    mut binning_resources: ResMut<StrandBinningBuffers>,
    mut shading_resources: ResMut<StrandShadingResources>,
    view_uniforms: Res<ViewUniforms>,
) {
    // This is an example of how to retrieve the shader storage buffer created in the main world above
    // and use it in the render world.
    for (entity, geometry) in query.iter() {
        // if raster_resources.bind_group.is_some() {
        //     continue;
        // }

        // --- Raster resources ---
        info!("Using strand geometry for entity: {:?}", entity);
        let Some(index_storage_buffer) = storage_buffers.get(&geometry.indices) else {
            warn!("Index storage buffer not found for entity: {:?}", entity);
            continue;
        };
        info!("[{:?}] Index storage buffer found.", entity);
        let Some(vertex_storage_buffer) = storage_buffers.get(&geometry.vertices) else {
            warn!("Vertex storage buffer not found for entity: {:?}", entity);
            continue;
        };
        let Some(meta_storage_buffer) = storage_buffers.get(&geometry.meta) else {
            warn!("Meta storage buffer not found for entity: {:?}", entity);
            continue;
        };
        let Some(geo_storage_buffer) = storage_buffers.get(&geometry.geos) else {
            warn!("Geo storage buffer not found for entity: {:?}", entity);
            continue;
        };

        info!(
            "[{:?}] Vertex storage buffer found: {:?}",
            entity, vertex_storage_buffer.buffer
        );
        let Some(froxel_buffer) = raster_resources.froxel_buffer.as_ref() else {
            warn!("Froxel buffer not found");
            continue;
        };
        let Some(output_texture) = raster_resources.output_texture.as_ref() else {
            warn!("Output texture not found");
            continue;
        };
        let Some(froxel_config_buffer) = raster_resources.froxel_config_buffer.as_ref() else {
            warn!("Froxel config buffer not found");
            continue;
        };

        let Some(tile_counts_buffer) = binning_resources.tile_counts_buffer.as_ref() else {
            warn!("Tile counts buffer not found");
            continue;
        };

        binning_resources.vertex_buffer = Some(vertex_storage_buffer.buffer.clone());
        binning_resources.meta_buffer = Some(meta_storage_buffer.buffer.clone());
        binning_resources.index_buffer = Some(index_storage_buffer.buffer.clone());
        binning_resources.geos_buffer = Some(geo_storage_buffer.buffer.clone());
        raster_resources.strand_count = Some(geometry.strand_count);

        let (shading_buffer, shading_buffer_view) = create_shading_target_texture(
            &device,
            geometry.strand_count,
            geometry.max_segments_in_strand,
        );
        shading_resources.output_texture = Some(shading_buffer_view);
        shading_resources.strand_count = Some(geometry.strand_count);
        shading_resources.max_segments_in_strand = Some(geometry.max_segments_in_strand);

        info!("Created bind group for strand rasterizer");
    }
}
use bevy::{
    ecs::world::World,
    log::*,
    pbr::{GlobalClusterableObjectMeta, LightMeta, ViewClusterBindings, ViewLightsUniformOffset},
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;

use crate::{
    pipelines::{
        binning::StrandBinningBuffers,
        prepass::{
            StrandPrepassPipeline, StrandPrepassResources, create_prepass_bind_group, run_prepass,
        },
        raster::StrandRasterizerResources,
    },
    resources::{ComputeInvocationDims, StochasticCullSettings},
};

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
        let binning_buffers = world.resource::<StrandBinningBuffers>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let invocation_dims = world.resource::<ComputeInvocationDims>();
        let cull_settings = world.resource::<StochasticCullSettings>();
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
        let Some(artifacts) = binning_buffers.artifacts.get(&view_entity) else {
            return Ok(());
        };
        let Some(froxel_config_buf) = raster_resources.froxel_config_buffer.get(&view_entity)
        else {
            return Ok(());
        };
        let Some(froxel_config) = raster_resources.frustrum_config.get(&view_entity) else {
            return Ok(());
        };
        render_context
            .command_encoder()
            .clear_buffer(&artifacts.tile_counts_buffer, 0, None);
        let tile_counts_binding = artifacts.tile_counts_buffer.as_entire_binding();
        let froxel_config_binding = froxel_config_buf.as_entire_binding();
        let tile_offsets_binding = artifacts.tile_offsets_buffer.as_entire_binding();
        let current_tile_write_indices_binding = artifacts
            .current_tile_write_indices_buffer
            .as_entire_binding();
        let froxel_tile_binding = artifacts.packed_segments_buffer.as_entire_binding();

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
            &tile_counts_binding,
            &froxel_config_binding,
            &tile_offsets_binding,
            &current_tile_write_indices_binding,
            &froxel_tile_binding,
        ) else {
            warn!("Failed to create prepass bind groups.");
            return Ok(());
        };
        let Some(indirect_args) = prepass_resources.indirect_args.as_ref() else {
            return Ok(());
        };
        debug!("Running prepass");
        run_prepass(
            render_context,
            pipeline_cache,
            prepass_pipeline,
            allocator,
            invocation_dims,
            froxel_config,
            cull_settings,
            &prepass_bind_group,
            &dynamic_offsets,
            indirect_args,
        );

        Ok(())
    }
}

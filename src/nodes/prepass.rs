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
use bevy_vsms::request::VirtualSurfaceRequestBitmapRuntime;

use crate::{
    pipelines::prepass::{
        StrandPrepassPipeline, StrandPrepassResources, create_prepass_bind_group, run_prepass,
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
        let request_runtime = world.resource::<VirtualSurfaceRequestBitmapRuntime>();
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
            request_runtime,
        ) else {
            warn!("Failed to create prepass bind groups.");
            return Ok(());
        };
        let Some(indirect_args) = prepass_resources.indirect_args.as_ref() else {
            return Ok(());
        };
        let Some(coarse_depth_lut) = prepass_resources.coarse_depth_lut.as_ref() else {
            return Ok(());
        };
        let Some(coarse_count_page_table) = prepass_resources.coarse_count_page_table.as_ref()
        else {
            return Ok(());
        };
        let Some(coarse_count_pages) = prepass_resources.coarse_count_pages.as_ref() else {
            return Ok(());
        };
        let Some(fine_cell_write_cursors) = prepass_resources.fine_cell_write_cursors.as_ref()
        else {
            return Ok(());
        };
        let Some(fine_seg_refs) = prepass_resources.fine_seg_refs.as_ref() else {
            return Ok(());
        };
        let Some(coarse_tile_work_counts) = prepass_resources.coarse_tile_work_counts.as_ref()
        else {
            return Ok(());
        };
        let Some(coarse_tile_work_offsets) = prepass_resources.coarse_tile_work_offsets.as_ref()
        else {
            return Ok(());
        };
        info!("Running prepass");
        run_prepass(
            render_context,
            pipeline_cache,
            prepass_pipeline,
            allocator,
            invocation_dims,
            cull_settings,
            &prepass_bind_group,
            &dynamic_offsets,
            indirect_args,
            coarse_depth_lut,
            coarse_count_page_table,
            coarse_count_pages,
            fine_cell_write_cursors,
            fine_seg_refs,
            coarse_tile_work_counts,
            coarse_tile_work_offsets,
            prepass_resources.frustum_count,
            prepass_resources.instance_count,
            prepass_resources.coarse_depth_tile_capacity,
            prepass_resources.coarse_range_capacity,
        );
        info!("Finished prepass");

        Ok(())
    }
}

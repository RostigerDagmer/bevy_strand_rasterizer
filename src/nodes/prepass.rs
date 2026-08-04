use bevy::{
    core_pipeline::prepass::ViewPrepassTextures,
    ecs::world::World,
    log::*,
    pbr::{GlobalClusterableObjectMeta, LightMeta, ViewClusterBindings, ViewLightsUniformOffset},
    prelude::Entity,
    render::{
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;
use bevy_vsms::request::VirtualSurfaceRequestBitmapRuntime;

use crate::{
    pipelines::prepass::{
        StrandPrepassPipeline, StrandPrepassResources, create_depth_reduce_bind_group,
        create_prepass_bind_groups, run_depth_reduce, run_prepass,
    },
    pipelines::raster::StrandRasterizerResources,
    resources::{
        ComputeInvocationDims, FineBinningBackend, PrepassTelemetrySettings, StochasticCullSettings,
    },
};

pub fn work_preparation_pass(
    world: &World,
    view: ViewQuery<Entity>,
    mut render_context: RenderContext,
) {
    let view_entity = view.entity();

    let pipeline_cache = world.resource::<PipelineCache>();
    let render_device = world.resource::<RenderDevice>();
    let allocator = world.resource::<GpuPagingAllocator>();
    let prepass_pipeline = world.resource::<StrandPrepassPipeline>();
    let prepass_resources = world.resource::<StrandPrepassResources>();
    let raster_resources = world.resource::<StrandRasterizerResources>();
    let request_runtime = world.resource::<VirtualSurfaceRequestBitmapRuntime>();
    let invocation_dims = world.resource::<ComputeInvocationDims>();
    let cull_settings = world.resource::<StochasticCullSettings>();
    let telemetry_settings = world.resource::<PrepassTelemetrySettings>();
    let fine_binning_backend = *world.resource::<FineBinningBackend>();
    let view_uniforms = world.resource::<ViewUniforms>();
    let light_meta = world.resource::<LightMeta>();
    let global_clusterable_object_meta = world.resource::<GlobalClusterableObjectMeta>();

    let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
        return;
    };

    let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity) else {
        return;
    };

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
        return;
    };

    let Some(clusterable_objects) = global_clusterable_object_meta
        .gpu_clustered_lights
        .binding()
    else {
        return;
    };

    let Some(view_cluster_bindings) = world.get::<ViewClusterBindings>(view_entity) else {
        return;
    };

    let Some(cluster_indices_binding) =
        view_cluster_bindings.clusterable_object_index_lists_binding()
    else {
        return;
    };

    let Some(cluster_offsets_binding) = view_cluster_bindings.offsets_and_counts_binding() else {
        return;
    };
    let Ok((
        prepass_bind_group,
        indirect_args_bind_group,
        prefix_indirect_args_bind_group,
        dynamic_offsets,
    )) = create_prepass_bind_groups(
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
    )
    else {
        warn!("Failed to create prepass bind groups.");
        return;
    };
    if let (Some(view_prepass_textures), Some(&frustum_id), Some(opaque_fine_depth_tiles)) = (
        world.get::<ViewPrepassTextures>(view_entity),
        raster_resources.frustum_ids.get(&view_entity),
        prepass_resources.opaque_fine_depth_tiles.as_ref(),
    ) {
        if let Some(config) = raster_resources.frustrum_config.get(&view_entity) {
            let fine_tiles_x = config.screen_width.div_ceil(config.froxel_size_x);
            let fine_tiles_y = config.screen_height.div_ceil(config.froxel_size_y);
            if let Ok(depth_reduce_bind_group) = create_depth_reduce_bind_group(
                render_device,
                prepass_pipeline,
                prepass_resources,
                view_prepass_textures,
            ) {
                run_depth_reduce(
                    &mut render_context,
                    pipeline_cache,
                    prepass_pipeline,
                    &depth_reduce_bind_group,
                    opaque_fine_depth_tiles,
                    frustum_id,
                    fine_tiles_x.saturating_mul(fine_tiles_y),
                );
            } else {
                render_context
                    .command_encoder()
                    .clear_buffer(opaque_fine_depth_tiles, 0, None);
            }
        }
    } else if let Some(opaque_fine_depth_tiles) = prepass_resources.opaque_fine_depth_tiles.as_ref()
    {
        render_context
            .command_encoder()
            .clear_buffer(opaque_fine_depth_tiles, 0, None);
    }
    let Some(indirect_args) = prepass_resources.indirect_args.as_ref() else {
        return;
    };
    let Some(prefix_indirect_args) = prepass_resources.prefix_indirect_args.as_ref() else {
        return;
    };
    let Some(telemetry) = prepass_resources.telemetry.as_ref() else {
        return;
    };
    let Some(coarse_depth_lut) = prepass_resources.coarse_depth_lut.as_ref() else {
        return;
    };
    let Some(coarse_count_page_table) = prepass_resources.coarse_count_page_table.as_ref() else {
        return;
    };
    let Some(coarse_count_pages) = prepass_resources.coarse_count_pages.as_ref() else {
        return;
    };
    let Some(fine_cell_write_cursors) = prepass_resources.fine_cell_write_cursors.as_ref() else {
        return;
    };
    let Some(page_candidate_counts) = prepass_resources.page_candidate_counts.as_ref() else {
        return;
    };
    let Some(page_candidate_cursors) = prepass_resources.page_candidate_cursors.as_ref() else {
        return;
    };
    let Some(virtual_page_candidate_counts) =
        prepass_resources.virtual_page_candidate_counts.as_ref()
    else {
        return;
    };
    let Some(fine_seg_refs) = prepass_resources.fine_seg_refs.as_ref() else {
        return;
    };
    info!("Running prepass");
    run_prepass(
        &mut render_context,
        pipeline_cache,
        prepass_pipeline,
        allocator,
        invocation_dims,
        cull_settings,
        &prepass_bind_group,
        &indirect_args_bind_group,
        &prefix_indirect_args_bind_group,
        &dynamic_offsets,
        indirect_args,
        prefix_indirect_args,
        telemetry,
        telemetry_settings.enabled,
        fine_binning_backend,
        coarse_depth_lut,
        coarse_count_page_table,
        coarse_count_pages,
        fine_cell_write_cursors,
        page_candidate_counts,
        page_candidate_cursors,
        virtual_page_candidate_counts,
        fine_seg_refs,
        prepass_resources.frustum_count,
        prepass_resources.instance_count,
        prepass_resources.max_strands_in_instance,
        prepass_resources.coarse_depth_tile_capacity,
        prepass_resources.coarse_count_page_capacity,
        prepass_resources.coarse_range_capacity,
    );
    info!("Finished prepass");
}

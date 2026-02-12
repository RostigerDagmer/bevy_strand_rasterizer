use bevy::{
    ecs::world::World,
    log::*,
    pbr::{
        GlobalClusterableObjectMeta, LightMeta, ShadowSamplers, ViewClusterBindings,
        ViewLightsUniformOffset, ViewShadowBindings,
    },
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniformOffset, ViewUniforms},
    },
};

use crate::pipelines::{
    raster::StrandRasterizerResources,
    shading::StrandShadingResources,
    shadows::{
        StrandShadowPipeline, StrandShadowResources, create_strand_shadow_bind_group,
        run_shadow_pass,
    },
};

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
        let shadow_resources = world.resource::<StrandShadowResources>();
        let shading_resources = world.resource::<StrandShadingResources>();
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
                // let Ok(strand_binning_bind_groups) = &create_strand_binning_bind_group(
                //     &entity,
                //     render_device,
                //     binning_pipeline,
                //     view_binding.clone(),
                //     view_uniform_offset,
                //     light_binding.clone(),
                //     view_light_uniform_offset,
                //     raster_resources,
                //     binning_buffers,
                // ) else {
                //     warn!("Failed to create strand binning bind group for shadows .");
                //     return Ok(());
                // };

                // run_binning_pass(
                //     &entity,
                //     render_device,
                //     render_context,
                //     pipeline_cache,
                //     strand_binning_bind_groups,
                //     binning_buffers,
                //     binning_pipeline,
                //     &frustrum,
                //     strand_count,
                //     true,
                // );

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

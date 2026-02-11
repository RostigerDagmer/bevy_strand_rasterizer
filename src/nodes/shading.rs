use bevy::{
    ecs::world::World,
    log::*,
    pbr::{
        GlobalClusterableObjectMeta, LightMeta, ShadowSamplers, ViewClusterBindings,
        ViewLightsUniformOffset,
    },
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniformOffset, ViewUniforms},
    },
};

use crate::pipelines::{
    binning::StrandBinningBuffers,
    raster::StrandRasterizerResources,
    shading::{
        StrandShadingPipeline, StrandShadingResources, create_strand_shading_bind_group,
        run_shading_pass,
    },
    shadows::StrandShadowResources,
};

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

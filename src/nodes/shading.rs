use bevy::{
    ecs::world::World,
    log::*,
    pbr::{LightMeta, ViewLightsUniformOffset},
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;

use crate::pipelines::{
    prepass::StrandPrepassResources,
    raster::StrandRasterizerResources,
    shading::{
        StrandShadingPipeline, StrandShadingResources, create_strand_shading_bind_group,
        run_shading_pass,
    },
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
        let prepass_resources = world.resource::<StrandPrepassResources>();
        let shading_resources = world.resource::<StrandShadingResources>();
        let allocator = world.resource::<GpuPagingAllocator>();
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
        // Shading bind group
        let Ok((shading_bind_group, shading_group_offsets)) = create_strand_shading_bind_group(
            render_device,
            &shading_pipeline,
            prepass_resources,
            &shading_resources,
            &view_binding,
            &light_binding,
            view_uniform_offset,
            view_light_uniform_offset,
        ) else {
            warn!("Failed to create strand shading bind group.");
            return Ok(());
        };

        run_shading_pass(
            render_context,
            pipeline_cache,
            shading_pipeline,
            prepass_resources,
            shading_resources,
            allocator,
            &shading_bind_group,
            &shading_group_offsets,
        );

        Ok(())
    }
}

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
    binning::StrandBinningBuffers,
    raster::{
        StrandRasterizerPipeline, StrandRasterizerResources, create_strand_raster_bind_group,
        run_raster_pass,
    },
    shading::StrandShadingResources,
};

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
        // let shading_pipeline = world.resource::<StrandShadingPipeline>();
        // let shadow_pipeline = world.resource::<StrandShadowPipeline>();
        let raster_pipeline = world.resource::<StrandRasterizerPipeline>();
        let allocator = world.resource::<GpuPagingAllocator>();
        let binning_buffers = world.resource::<StrandBinningBuffers>();
        let shading_resources = world.resource::<StrandShadingResources>();
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

        // Raster bind group
        let Ok((raster_bind_group, raster_group_offsets)) = create_strand_raster_bind_group(
            &view_entity,
            render_device,
            &raster_pipeline,
            &raster_resources,
            &binning_buffers,
            &view_binding,
            &light_binding,
            &view_uniform_offset,
            &view_light_uniform_offset,
        ) else {
            warn!("Failed to create strand raster bind group.");
            return Ok(());
        };

        run_raster_pass(
            render_context,
            pipeline_cache,
            raster_pipeline,
            allocator,
            &frustrum,
            raster_resources,
            shading_resources,
            &raster_bind_group,
            &raster_group_offsets,
        );

        Ok(())
    }
}

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

        let view_entity = graph.view_entity();
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let allocator = world.resource::<GpuPagingAllocator>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let prepass_resources = world.resource::<StrandPrepassResources>();
        let shadow_pipeline = world.resource::<StrandShadowPipeline>();
        let shadow_resources = world.resource::<StrandShadowResources>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let light_meta = world.resource::<LightMeta>();

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

        let Ok((shadow_bind_group, dynamic_offsets)) = create_strand_shadow_bind_group(
            render_device,
            shadow_pipeline,
            shadow_resources,
            prepass_resources,
            &view_binding,
            &light_binding,
            view_uniform_offset,
            view_light_uniform_offset,
        ) else {
            warn!("Failed to create strand shadow bind group.");
            return Ok(());
        };

        let mut frusta: Vec<_> = raster_resources.frustrum_config.iter().collect();
        frusta.sort_by_key(|(entity, _)| entity.index());
        for (entity, frustum_cfg) in frusta {
            let Some(&frustum_id) = raster_resources.frustum_ids.get(entity) else {
                continue;
            };
            if !shadow_resources
                .light_layer_by_frustum
                .contains_key(&frustum_id)
            {
                continue;
            }
            run_shadow_pass(
                render_context,
                pipeline_cache,
                shadow_pipeline,
                allocator,
                frustum_cfg,
                frustum_id,
                raster_resources,
                &shadow_bind_group,
                &dynamic_offsets,
            );
        }

        Ok(())
    }
}

use bevy::{
    ecs::world::World,
    log::*,
    pbr::{LightMeta, ViewLightsUniformOffset, ViewShadowBindings},
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice},
        view::{ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;
use bevy_vsms::{allocator::VirtualSurfaceRuntime, api::VirtualSurfaceKind};

use crate::pipelines::{
    prepass::StrandPrepassResources,
    raster::StrandRasterizerResources,
    shadows::{
        StrandShadowPipeline, create_strand_shadow_bind_group, create_vsms_table_bind_group,
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
        let vsms_runtime = world.resource::<VirtualSurfaceRuntime>();
        let raster_resources = world.resource::<StrandRasterizerResources>();
        let prepass_resources = world.resource::<StrandPrepassResources>();
        let shadow_pipeline = world.resource::<StrandShadowPipeline>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let light_meta = world.resource::<LightMeta>();

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            return Ok(());
        };
        let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity)
        else {
            return Ok(());
        };
        let Some(view_shadow_bindings) = world.get::<ViewShadowBindings>(view_entity) else {
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
            prepass_resources,
            &view_binding,
            &light_binding,
            &view_shadow_bindings.directional_light_depth_texture_view,
            view_uniform_offset,
            view_light_uniform_offset,
        ) else {
            warn!("Failed to create strand shadow bind group.");
            return Ok(());
        };
        let Some(opacity_storage_bind_group) = vsms_runtime
            .pool_storage_bindings
            .get(&VirtualSurfaceKind::Opacity3D)
            .map(|b| &b.bind_group)
        else {
            return Ok(());
        };
        let Some(depth_storage_bind_group) = vsms_runtime
            .pool_storage_bindings
            .get(&VirtualSurfaceKind::Depth2DArray)
            .map(|b| &b.bind_group)
        else {
            return Ok(());
        };
        let Some(opacity_pool) = vsms_runtime.pools.get(&VirtualSurfaceKind::Opacity3D) else {
            return Ok(());
        };
        let Some(depth_pool) = vsms_runtime.pools.get(&VirtualSurfaceKind::Depth2DArray) else {
            return Ok(());
        };
        let Some(opacity_table_bind_group) = create_vsms_table_bind_group(
            render_device,
            &shadow_pipeline.vsms_table_bind_group_layout,
            opacity_pool,
            "strand_shadow_opacity_vsms_table_bind_group",
        ) else {
            return Ok(());
        };
        let Some(depth_table_bind_group) = create_vsms_table_bind_group(
            render_device,
            &shadow_pipeline.vsms_table_bind_group_layout,
            depth_pool,
            "strand_shadow_depth_vsms_table_bind_group",
        ) else {
            return Ok(());
        };
        let Some(raster_tile_run_dispatch_args) =
            prepass_resources.raster_tile_run_dispatch_args.as_ref()
        else {
            return Ok(());
        };

        run_shadow_pass(
            render_context,
            pipeline_cache,
            shadow_pipeline,
            allocator,
            opacity_storage_bind_group,
            depth_storage_bind_group,
            &opacity_table_bind_group,
            &depth_table_bind_group,
            raster_resources,
            prepass_resources.frustum_count,
            raster_tile_run_dispatch_args,
            &shadow_bind_group,
            &dynamic_offsets,
        );

        Ok(())
    }
}

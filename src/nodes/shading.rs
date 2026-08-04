use bevy::{
    ecs::world::World,
    log::*,
    pbr::{LightMeta, ViewLightsUniformOffset},
    prelude::Entity,
    render::{
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;
use bevy_vsms::{allocator::VirtualSurfaceRuntime, api::VirtualSurfaceKind};

use crate::pipelines::{
    prepass::StrandPrepassResources,
    raster::StrandRasterizerResources,
    shading::{
        StrandShadingPipeline, StrandShadingResources, create_strand_shading_bind_group,
        run_shading_pass,
    },
    shadows::{StrandShadowPipeline, create_vsms_table_bind_group},
};

pub fn strand_shading_pass(
    world: &World,
    view: ViewQuery<Entity>,
    mut render_context: RenderContext,
) {
    // Check if we have resources
    if !world.contains_resource::<StrandRasterizerResources>() {
        return;
    }
    let view_entity = view.entity(); // Get the entity this pass instance is running for

    let pipeline_cache = world.resource::<PipelineCache>();
    let render_device = world.resource::<RenderDevice>();
    let shading_pipeline = world.resource::<StrandShadingPipeline>();
    let prepass_resources = world.resource::<StrandPrepassResources>();
    let shading_resources = world.resource::<StrandShadingResources>();
    let allocator = world.resource::<GpuPagingAllocator>();
    let vsms_runtime = world.resource::<VirtualSurfaceRuntime>();
    let shadow_pipeline = world.resource::<StrandShadowPipeline>();
    let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms
    let light_meta = world.resource::<LightMeta>(); // Get light meta

    let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
        // This node might run on views without this (e.g. shadow maps). Handle appropriately.
        warn!(
            "Node running on view {:?} without ViewUniformOffset",
            view_entity
        );
        return;
    };

    let Some(view_light_uniform_offset) = world.get::<ViewLightsUniformOffset>(view_entity) else {
        // This node might run on views without this (e.g. shadow maps). Handle appropriately.
        warn!(
            "Node running on view {:?} without ViewLightUniformOffset",
            view_entity
        );
        return;
    };

    // --- Check Prerequisites ---
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        warn!("ViewUniforms binding not available.");
        return;
    };

    let Some(light_binding) = light_meta.view_gpu_lights.binding() else {
        // This node might run on views without this (e.g. shadow maps). Handle appropriately.
        warn!(
            "Node running on view {:?} without LightBinding",
            view_entity
        );
        return;
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
        return;
    };
    let Some(opacity_pool_bind_group) = vsms_runtime
        .pool_bindings
        .get(&VirtualSurfaceKind::Opacity3D)
        .map(|b| &b.bind_group)
    else {
        return;
    };
    let Some(depth_pool_bind_group) = vsms_runtime
        .pool_bindings
        .get(&VirtualSurfaceKind::Depth2DArray)
        .map(|b| &b.bind_group)
    else {
        return;
    };
    let Some(opacity_pool) = vsms_runtime.pools.get(&VirtualSurfaceKind::Opacity3D) else {
        return;
    };
    let Some(depth_pool) = vsms_runtime.pools.get(&VirtualSurfaceKind::Depth2DArray) else {
        return;
    };
    let Some(opacity_table_bind_group) = create_vsms_table_bind_group(
        render_device,
        &shadow_pipeline.vsms_table_bind_group_layout,
        opacity_pool,
        "strand_shading_opacity_vsms_table_bind_group",
    ) else {
        return;
    };
    let Some(depth_table_bind_group) = create_vsms_table_bind_group(
        render_device,
        &shadow_pipeline.vsms_table_bind_group_layout,
        depth_pool,
        "strand_shading_depth_vsms_table_bind_group",
    ) else {
        return;
    };

    run_shading_pass(
        &mut render_context,
        pipeline_cache,
        shading_pipeline,
        prepass_resources,
        shading_resources,
        allocator,
        &shading_bind_group,
        opacity_pool_bind_group,
        depth_pool_bind_group,
        &opacity_table_bind_group,
        &depth_table_bind_group,
        &shading_group_offsets,
    );
}

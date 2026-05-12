use bevy::{
    ecs::world::World,
    log::*,
    pbr::{
        LightEntity, LightMeta, ShadowView, ViewLightEntities, ViewLightsUniformOffset,
        ViewShadowBindings,
    },
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{ImageSubresourceRange, PipelineCache, TextureAspect},
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewUniformOffset, ViewUniforms},
    },
};
use bevy_gpu_paging_allocator::GpuPagingAllocator;
use bevy_vsms::{allocator::VirtualSurfaceRuntime, api::VirtualSurfaceKind};

use crate::pipelines::{
    prepass::StrandPrepassResources,
    raster::StrandRasterizerResources,
    shadows::{
        ShadowStampParams, StrandShadowPipeline, StrandShadowResources,
        create_shadow_stamp_bind_group, create_strand_shadow_bind_group,
        create_vsms_table_bind_group, run_shadow_pass, run_shadow_stampback_pass,
    },
};
use wgpu::util::BufferInitDescriptor;

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
        let Some(view_shadow_bindings) = world.get::<ViewShadowBindings>(view_entity) else {
            return Ok(());
        };
        let Some(view_light_entities) = world.get::<ViewLightEntities>(view_entity) else {
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
        let Some(depth_sample_bind_group) = vsms_runtime
            .pool_bindings
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

        let clear_range = ImageSubresourceRange {
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        };
        for surface in &depth_pool.physical_surfaces {
            render_context
                .command_encoder()
                .clear_texture(&surface.texture, &clear_range);
        }

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

        for target in shadow_resources.stamp_targets.iter().copied() {
            if target.frustum_id >= prepass_resources.frustum_count {
                continue;
            }
            let Some((shadow_view, extracted_view)) =
                view_light_entities.lights.iter().find_map(|entity| {
                    let light_entity = world.get::<LightEntity>(*entity)?;
                    let LightEntity::Directional {
                        light_entity,
                        cascade_index,
                    } = light_entity
                    else {
                        return None;
                    };
                    if *light_entity == target.light_entity
                        && *cascade_index as u32 == target.cascade_index
                    {
                        Some((
                            world.get::<ShadowView>(*entity)?,
                            world.get::<ExtractedView>(*entity)?,
                        ))
                    } else {
                        None
                    }
                })
            else {
                continue;
            };
            let params = ShadowStampParams {
                frustum_id: target.frustum_id,
                target_width: extracted_view.viewport.z,
                target_height: extracted_view.viewport.w,
                _pad2: 0,
            };
            let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("strand_shadow_stampback_params"),
                contents: bytemuck::bytes_of(&params),
                usage: bevy::render::render_resource::BufferUsages::UNIFORM,
            });
            let Some(stamp_bind_group) = create_shadow_stamp_bind_group(
                render_device,
                shadow_pipeline,
                prepass_resources,
                &params_buffer,
            ) else {
                continue;
            };
            run_shadow_stampback_pass(
                render_context,
                pipeline_cache,
                shadow_pipeline,
                depth_sample_bind_group,
                &depth_table_bind_group,
                &stamp_bind_group,
                &shadow_view.depth_attachment.view,
            );
        }

        Ok(())
    }
}

use bevy::{
    ecs::world::World,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::RenderContext,
        view::ViewTarget,
    },
};
use wgpu::{
    BindGroupEntry, BindingResource, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, util::BufferInitDescriptor,
};

use crate::{
    pipelines::{
        layouts,
        prepass::StrandPrepassResources,
        raster::StrandRasterizerResources,
        tile_debug::{TileDebugParams, TileDebugPipeline},
    },
    resources::TileDebugSettings,
};

#[derive(Default)]
pub struct TileDebugNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct TileDebugLabel;

impl Node for TileDebugNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let settings = world.resource::<TileDebugSettings>();
        if !settings.enabled {
            return Ok(());
        }

        let view_entity = graph.view_entity();
        let Some(view_target) = world.get::<ViewTarget>(view_entity) else {
            return Ok(());
        };

        let raster_resources = world.resource::<StrandRasterizerResources>();
        let prepass_resources = world.resource::<StrandPrepassResources>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<TileDebugPipeline>();

        let Some(frustum_table) = prepass_resources.frustum_table.as_ref() else {
            return Ok(());
        };
        let Some(froxel_bucket_heads) = prepass_resources.froxel_bucket_heads.as_ref() else {
            return Ok(());
        };
        let Some(chunk_pool) = prepass_resources.chunk_pool.as_ref() else {
            return Ok(());
        };
        let Some(coarse_count_page_table) = prepass_resources.coarse_count_page_table.as_ref()
        else {
            return Ok(());
        };
        let Some(coarse_count_pages) = prepass_resources.coarse_count_pages.as_ref() else {
            return Ok(());
        };

        let mut frusta: Vec<_> = raster_resources.frustrum_config.keys().copied().collect();
        frusta.sort_by_key(|e| e.index());
        let Some(frustum_id) = frusta
            .iter()
            .position(|entity| *entity == view_entity)
            .map(|idx| idx as u32)
        else {
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let norm = raster_resources.strand_count.unwrap_or(1).max(1) as f32;
        let params = TileDebugParams {
            inv_norm: 1.0 / norm,
            gain: settings.gain,
            alpha: settings.alpha,
            frustum_id,
        };
        let params_buffer =
            render_context
                .render_device()
                .create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("tile_debug_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: bevy::render::render_resource::BufferUsages::UNIFORM,
                });

        let post_process = view_target.post_process_write();
        let bind_group = render_context.render_device().create_bind_group(
            "tile_debug_bind_group",
            &pipeline_res.layout,
            &[
                BindGroupEntry {
                    binding: layouts::tile_debug::FRUSTUM_TABLE,
                    resource: frustum_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::FROXEL_BUCKET_HEADS,
                    resource: froxel_bucket_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::CHUNK_POOL,
                    resource: chunk_pool.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::COARSE_COUNT_PAGE_TABLE,
                    resource: coarse_count_page_table.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::COARSE_COUNT_PAGES,
                    resource: coarse_count_pages.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::PARAMS,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::INPUT_TEXTURE,
                    resource: BindingResource::TextureView(post_process.source),
                },
                BindGroupEntry {
                    binding: layouts::tile_debug::INPUT_SAMPLER,
                    resource: BindingResource::Sampler(&pipeline_res.sampler),
                },
            ],
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("tile_debug_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

use bevy::{
    core_pipeline::prepass::ViewPrepassTextures,
    ecs::world::World,
    log::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::PipelineCache,
        renderer::RenderContext,
        view::{ViewTarget, ViewUniformOffset, ViewUniforms},
    },
};
use wgpu::{
    BindGroupEntry, BindingResource, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp,
};

use crate::pipelines::{composite::CompositionPipeline, raster::StrandRasterizerResources};

#[derive(Debug, Clone, Default)]
pub struct DomPunchThroughNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct DomPunchThroughLabel;

impl Node for DomPunchThroughNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // pass
        Ok(())
    }
}

#[derive(Default)]
pub struct CompositionNode;

#[derive(Debug, Clone, Hash, PartialEq, Eq, RenderLabel)]
pub struct CompositionLabel;

impl Node for CompositionNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();
        let view_uniforms = world.resource::<ViewUniforms>(); // Get current view uniforms

        let Some(view_target) = world.get::<ViewTarget>(view_entity) else {
            // This can happen if the view doesn't have a ViewTarget
            // (e.g., shadow map views, reflection probes)
            debug!("View entity {:?} does not have a ViewTarget", view_entity);
            return Ok(());
        };

        let Some(view_uniform_offset) = world.get::<ViewUniformOffset>(view_entity) else {
            // This node might run on views without this (e.g. shadow maps). Handle appropriately.
            warn!(
                "Node running on view {:?} without ViewUniformOffset",
                view_entity
            );
            return Ok(());
        };

        let Some(depth_target) = world.get::<ViewPrepassTextures>(view_entity) else {
            warn!("View entity {:?} does not have a DepthTarget", view_entity);
            return Ok(());
        };
        let Some(composition_pipeline) = world.get_resource::<CompositionPipeline>() else {
            warn!("CompositionPipeline not found");
            return Ok(());
        };
        let Some(strand_raster_resources) = world.get_resource::<StrandRasterizerResources>()
        else {
            warn!("StrandRasterizerResources not found");
            return Ok(());
        };
        let Some(strand_output_texture) = strand_raster_resources.output_texture.as_ref() else {
            warn!("Strand output texture not ready");
            return Ok(());
        };
        let Some(strand_depth_texture) = strand_raster_resources.output_depth.as_ref() else {
            warn!("Strand depth texture not ready");
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(composition_pipeline.pipeline)
        else {
            warn!("Composition render pipeline not ready (pipeline cache)");
            return Ok(());
        };

        // Get the input texture (result of main pass)
        // let input_texture = view_target.get_color_attachment().view; // <- this one is multisampled
        let input_texture = view_target.main_texture_view();
        debug!("Depth target {:?}", depth_target.depth_view());
        let scene_depth_texture = depth_target.depth_view().expect("Depth prepass enabled");

        let bind_group = render_context.render_device().create_bind_group(
            "composition_bind_group",
            &composition_pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: view_uniforms.uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(input_texture), // Main scene texture
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&composition_pipeline.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(strand_output_texture), // Strand texture
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(strand_depth_texture), // Strand depth texture
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(scene_depth_texture), // Scene depth texture
                },
            ],
        );

        // Use the ViewTarget's post_process_write to get the correct target
        let post_process = view_target.post_process_write();
        let destination_texture = post_process.destination; // TextureView to write to

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("composition_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: destination_texture, // Write to the destination
                resolve_target: None,
                ops: Operations {
                    // Load the existing contents (result of main pass, potentially clear if first post-proc)
                    // If this is the *first* post-processing pass Bevy runs, it might
                    // have cleared it already. If it runs *after* other passes, load.
                    // Load is usually safer.
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
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1); // Draw a fullscreen triangle
        info!("Finished Compositing");
        Ok(())
    }
}

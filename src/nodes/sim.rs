use bevy::{
    ecs::world::World,
    prelude::Entity,
    render::{
        render_resource::PipelineCache,
        renderer::{RenderContext, RenderDevice, ViewQuery},
    },
};

use crate::pipelines::sim::StrandSimulatorResources;

pub fn strand_simulation_pass(
    world: &World,
    view: ViewQuery<Entity>,
    mut _render_context: RenderContext,
) {
    // Check if we have resources
    if !world.contains_resource::<StrandSimulatorResources>() {
        return;
    }
    let _view_entity = view.entity(); // Get the entity this pass instance is running for

    let _pipeline_cache = world.resource::<PipelineCache>();
    let _render_device = world.resource::<RenderDevice>();
}

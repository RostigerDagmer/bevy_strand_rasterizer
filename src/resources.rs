use bevy::{prelude::*, render::render_resource::BindGroup, utils::HashMap};
use crate::shader_types::PushConstants;

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    instances: HashMap<Entity, StrandAssetInstance>,
}

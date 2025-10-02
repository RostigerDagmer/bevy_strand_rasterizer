use crate::shader_types::PushConstants;
use bevy::{prelude::*, render::render_resource::BindGroup};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StrandAssetInstance {
    push_constants: PushConstants,
    bind_group: BindGroup,
}

#[derive(Resource, Default)]
pub struct StrandAssetResources {
    instances: HashMap<Entity, StrandAssetInstance>,
}

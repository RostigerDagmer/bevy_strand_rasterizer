use bevy::pbr::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
mod dson;
use dson::*;
mod components;
use components::*;
mod resources;
use plugin::StrandRasterizerPlugin;
mod pipelines;
mod plugin;
mod shader_types;
mod perf;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let handle: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    commands.spawn((StrandAsset { handle }));

    commands.spawn((
        FroxelConfig::default(),
        Transform::from_xyz(-1.0, 7., 1.0).looking_at(Vec3::new(-1.0, 1., 0.), Vec3::Y),
        PanOrbitCamera::default(),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.05, 0.05, 0.05))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    // add one light
    // commands.spawn((
    //     PointLight {
    //         shadows_enabled: true,
    //         ..default()
    //     },
    //     Transform::from_xyz(4.0, 8.0, 4.0),
    // ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(2.0, 6.0, 0.0),
            rotation: Quat::from_rotation_x(-3.141592 / 4.),
            ..default()
        },
        FroxelConfig { // TODO: move into plugin
            screen_width: 2048, // shadow map size in this context
            screen_height: 2048,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 32,
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 10.0,
            maximum_distance: 15.0,
            ..default()
        }
        .build(),
    ));
}

fn debug_print_geo(query: Query<&StrandAsset>, assets: Res<Assets<DsonAsset>>) {
    for st_asset in query.iter() {
        let Some(asset) = assets.get(&st_asset.handle) else {
            continue;
        };
        info!(
            "geo: {:?}",
            asset.dson_file.geometry_library.as_ref().unwrap()[0].polyline_list
        );
    }
}

const CULL_MIN_DIST: f32 = 4.0; // TODO: set this to the distance of the farthest object
const CULL_MAX_DIST: f32 = 100.0; // TODO: set this to the distance of the closest object

fn percent_culled(distance: f32) -> f32 {
    let norm_distance = (distance - CULL_MIN_DIST) / CULL_MAX_DIST;
    norm_distance.powf(0.5)
}

fn debug_print_camera_distance_to_aabb(query: Query<&StrandGeometry>, camera: Query<(&Camera, &Transform)>) {
    for strand_geo in query.iter() {
        let aabb = strand_geo.aabb;
        let aabb_center = aabb.max - aabb.min;
        for (c, transform) in camera.iter() {
            let dist = aabb_center.distance(transform.translation.into());
            info!("Distance to camera: {:?} -> percent after cull: {:?}", dist, percent_culled(dist));
        }
    }

}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(StrandRasterizerPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(perf::FpsDisplayPlugin)
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .add_systems(Startup, setup)
        // .add_systems(Update, debug_print_geo)
        // .add_systems(Update, debug_print_camera_distance_to_aabb)
        .run();
}

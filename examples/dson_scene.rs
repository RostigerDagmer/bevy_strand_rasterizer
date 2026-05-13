mod common;

use bevy::prelude::*;
use default_material::DefaultMaterial;
use strand_software_rasterizer::prelude::*;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut default_materials: ResMut<Assets<DefaultMaterial>>,
) {
    let curly_hair = asset_server.load("curly/curly.dsf");
    commands.spawn((
        StrandAsset { handle: curly_hair },
        StrandMaterial {
            absorption_color: Vec4::new(0.55, 0.33, 0.198, 0.55),
            specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
            ambient_factor: 0.15,
            ao_factor: 0.1,
            eta: 1.55,
            beta: 0.85,
            alpha: 0.35,
            shift: 0.1,
            min_radius_pixels: 1.0,
            max_radius_pixels: 1.5,
        },
        Transform::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    let pixie_cut = asset_server.load("dForce Pixie Cut_708408.dsf");
    commands.spawn((
        StrandAsset { handle: pixie_cut },
        StrandMaterial {
            absorption_color: Vec4::new(0.432, 0.224, 0.133, 0.3),
            specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
            ambient_factor: 0.15,
            ao_factor: 0.1,
            eta: 1.55,
            beta: 0.85,
            alpha: 0.35,
            shift: 0.1,
            min_radius_pixels: 0.7,
            max_radius_pixels: 2.0,
        },
        Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    common::spawn_camera(
        &mut commands,
        Transform::from_xyz(-2.0, 2.5, 4.0).looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
        Vec3::new(0.0, 0.8, 0.0),
    );
    common::spawn_floor_and_light(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut default_materials,
        0.0,
    );
}

fn main() {
    let mut app = App::new();
    common::add_example_plugins(&mut app)
        .add_systems(Startup, setup)
        .add_systems(Update, common::rotate_light_around_z_axis)
        .run();
}

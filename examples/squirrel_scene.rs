mod common;

use bevy::{
    animation::graph::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex},
    gltf::GltfAssetLabel,
    prelude::*,
};
use default_material::DefaultMaterial;
use strand_software_rasterizer::prelude::*;

const SQUIRREL_SPLIT_CACHE_DIR: &str = "assets/squirrel/RedSquirrel_Winter_1.strands.d";
const SQUIRREL_SPLIT_ASSET_DIR: &str = "squirrel/RedSquirrel_Winter_1.strands.d";
const SQUIRREL_MONOLITHIC_CACHE: &str = "assets/squirrel/RedSquirrel_Winter_1.strands";
const SQUIRREL_MONOLITHIC_ASSET: &str = "squirrel/RedSquirrel_Winter_1.strands";
const SQUIRREL_GLTF: &str = "squirrel/gltf/RedSquirrel_Winter_1.gltf";

#[derive(Resource)]
struct SquirrelPoseAnimation {
    graph: Handle<AnimationGraph>,
    node: AnimationNodeIndex,
}

fn squirrel_world_transform() -> Transform {
    Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)).with_scale(Vec3::splat(8.0))
}

fn squirrel_strand_material(group_name: &str) -> StrandMaterial {
    let mut material = StrandMaterial {
        absorption_color: Vec4::new(0.56, 0.25, 0.14, 0.45),
        specular_color: Vec4::new(0.9, 0.72, 0.52, 0.2),
        ambient_factor: 0.15,
        ao_factor: 0.1,
        eta: 1.65,
        beta: 0.85,
        alpha: 0.35,
        shift: 0.1,
        min_radius_pixels: 0.3,
        max_radius_pixels: 1.4,
    };

    if group_name.contains("_Tail") {
        material.absorption_color = Vec4::new(0.68, 0.32, 0.18, 0.5);
        material.specular_color = Vec4::new(0.98, 0.72, 0.64, 0.12);
        material.max_radius_pixels = 1.8;
    } else if group_name.contains("_Beard") || group_name.contains("_Eye")
    // || group_name.contains("Head")
    {
        material.absorption_color = Vec4::new(0.85, 0.72, 0.56, 0.45);
        material.specular_color = Vec4::new(0.98, 0.72, 0.64, 0.2);
        material.beta = 0.75;
        material.alpha = 0.25;
        material.max_radius_pixels = 1.2;
    }

    material
}

fn spawn_squirrel_strand_group(commands: &mut Commands, asset_server: &AssetServer) -> bool {
    let split_dir = std::path::Path::new(SQUIRREL_SPLIT_CACHE_DIR);
    if split_dir.is_dir() {
        let Ok(entries) = std::fs::read_dir(split_dir) else {
            return false;
        };
        let mut cache_files: Vec<_> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "strands"))
            .collect();
        cache_files.sort();

        if cache_files.is_empty() {
            return false;
        }

        commands
            .spawn((
                Name::new("RedSquirrel_Winter_1 strands"),
                squirrel_world_transform(),
                GlobalTransform::default(),
            ))
            .with_children(|parent| {
                for path in cache_files {
                    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                        continue;
                    };
                    parent.spawn((
                        Name::new(file_name.to_string()),
                        StrandCache {
                            handle: asset_server
                                .load(format!("{SQUIRREL_SPLIT_ASSET_DIR}/{file_name}")),
                        },
                        squirrel_strand_material(file_name),
                        Transform::default(),
                        GlobalTransform::default(),
                    ));
                }
            });
        return true;
    }

    if std::path::Path::new(SQUIRREL_MONOLITHIC_CACHE).exists() {
        commands.spawn((
            Name::new("RedSquirrel_Winter_1 strands"),
            StrandCache {
                handle: asset_server.load(SQUIRREL_MONOLITHIC_ASSET),
            },
            squirrel_strand_material("RedSquirrel_Winter_1"),
            squirrel_world_transform(),
            GlobalTransform::default(),
        ));
        return true;
    }

    false
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut default_materials: ResMut<Assets<DefaultMaterial>>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
) {
    let (squirrel_pose_graph, squirrel_pose_node) = AnimationGraph::from_clip(
        asset_server.load(GltfAssetLabel::Animation(0).from_asset(SQUIRREL_GLTF)),
    );
    commands.insert_resource(SquirrelPoseAnimation {
        graph: animation_graphs.add(squirrel_pose_graph),
        node: squirrel_pose_node,
    });

    commands.spawn((
        Name::new("RedSquirrel_Winter_1 base geometry"),
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(SQUIRREL_GLTF))),
        squirrel_world_transform(),
        GlobalTransform::default(),
    ));
    spawn_squirrel_strand_group(&mut commands, &asset_server);

    common::spawn_camera(
        &mut commands,
        Transform::from_xyz(-2.0, 4.5, 0.0).looking_at(Vec3::new(-1.0, 1.0, 0.0), Vec3::Y),
        Vec3::new(0.0, 4.1, 0.0),
    );
    common::spawn_floor_and_light(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut default_materials,
        2.0,
    );
}

fn apply_squirrel_pose_animation(
    mut commands: Commands,
    pose_animation: Option<Res<SquirrelPoseAnimation>>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    let Some(pose_animation) = pose_animation else {
        return;
    };

    for (entity, mut player) in &mut players {
        player.start(pose_animation.node).set_seek_time(0.0).pause();
        commands
            .entity(entity)
            .insert(AnimationGraphHandle(pose_animation.graph.clone()));
    }
}

fn main() {
    let mut app = App::new();
    common::add_example_plugins(&mut app)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                apply_squirrel_pose_animation,
                common::rotate_light_around_z_axis,
            ),
        )
        .run();
}

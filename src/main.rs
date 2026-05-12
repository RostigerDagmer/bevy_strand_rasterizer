use bevy::{
    animation::graph::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex},
    gltf::GltfAssetLabel,
    light::CascadeShadowConfigBuilder,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_vsms::prelude::BevyVsmsPlugin;
mod dson;
use default_material::{DefaultMaterial, DefaultMaterialPlugin, make_default_material};
use dson::*;
mod components;
use components::*;
mod nodes;
mod resources;
use plugin::StrandRasterizerPlugin;
use resources::{StochasticCullSettings, TileDebugSettings};
mod strand_cache;
use strand_cache::*;
mod perf;
mod pipelines;
mod plugin;
mod shader_types;
use bevy_gpu_paging_allocator as allocator;
use wgpu::Backends;

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
        absorption_color: Vec4::new(0.42, 0.25, 0.14, 0.35),
        specular_color: Vec4::new(0.9, 0.72, 0.52, 0.15),
        ambient_factor: 0.15,
        ao_factor: 0.1,
        eta: 1.55,
        beta: 0.85,
        alpha: 0.35,
        shift: 0.1,
        min_radius_pixels: 0.7,
        max_radius_pixels: 1.3,
    };

    if group_name.contains("_Tail") {
        material.absorption_color = Vec4::new(0.5, 0.32, 0.18, 0.6);
        material.min_radius_pixels = 1.0;
        material.max_radius_pixels = 1.6;
    } else if group_name.contains("_Beard") || group_name.contains("_Eye") {
        material.absorption_color = Vec4::new(0.85, 0.72, 0.56, 0.45);
        material.max_radius_pixels = 1.0;
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
                    // if file_name.contains("Body")
                    // || file_name.contains("Claw")
                    // || file_name.contains("Tail")
                    // || file_name.contains("Ears")
                    // || file_name.contains("Eye")
                    // {
                    let asset_path = format!("{SQUIRREL_SPLIT_ASSET_DIR}/{file_name}");
                    parent.spawn((
                        Name::new(file_name.to_string()),
                        StrandCache {
                            handle: asset_server.load(asset_path),
                        },
                        squirrel_strand_material(file_name),
                        Transform::default(),
                        GlobalTransform::default(),
                    ));
                    // }
                }
            });
        return true;
    }

    if std::path::Path::new(SQUIRREL_MONOLITHIC_CACHE).exists() {
        let handle: Handle<StrandCacheAsset> =
            asset_server.load(SQUIRREL_MONOLITHIC_ASSET.to_string());
        commands.spawn((
            Name::new("RedSquirrel_Winter_1 strands"),
            StrandCache { handle },
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
    mut extended_materials: ResMut<Assets<DefaultMaterial>>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
) {
    // let handle2: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    // let handle: Handle<DsonAsset> = asset_server.load("curly/curly.dsf".to_string());
    // // let handle: Handle<DsonAsset> = asset_server.load("Wet Messy Bang Hair/Wet Messy Bang Hair.dsf".to_string());
    // commands.spawn((
    //     StrandAsset { handle },
    //     StrandMaterial {
    //         // absorption_color: Vec4::new(0.7, 0.7, 0.7, 0.5),
    //         // specular_color: Vec4::new(0.7, 0.7, 0.7, 0.1),

    //         // absorption_color: Vec4::new(0.6, 0.1, 0.05, 0.3),
    //         // absorption_color: Vec4::new(0.52, 0.13, 0.1, 0.3),
    //         // absorption_color: Vec4::new(0.432, 0.224, 0.133, 0.3),
    //         // absorption_color: Vec4::new(1.0, 209.0 / 255.0, 184.0 / 255.0, 0.8),
    //         absorption_color: Vec4::new(0.55, 0.33, 0.198, 0.55),
    //         // specular_color: Vec4::new(0.93, 0.48, 0.375, 2.0),
    //         // absorption_color: Vec4::new(0.55, 0.38, 0.07, 0.4),
    //         // specular_color: Vec4::new(0.87, 0.7, 0.38, 0.6),
    //         // specular_color: Vec4::new(0.82, 0.663, 0.365, 0.3),
    //         specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
    //         // specular_color: Vec4::new(0.769, 0.383, 0.325, 0.9),
    //         ambient_factor: 0.15,
    //         ao_factor: 0.1,
    //         eta: 1.55, // index of refraction
    //         // beta: 0.25,   // higher order path roughness
    //         // alpha: 0.05, // first order path roughness
    //         // beta: 0.45,
    //         beta: 0.85,
    //         alpha: 0.35,
    //         shift: 0.1, // specular shift
    //         min_radius_pixels: 1.0,
    //         max_radius_pixels: 1.5,
    //     },
    //     // StrandMaterial::default(),
    //     Transform::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
    //     GlobalTransform::default(),
    // ));

    // commands.spawn((
    //     StrandAsset { handle: handle2 },
    //     StrandMaterial {
    //         // absorption_color: Vec4::new(0.7, 0.7, 0.7, 0.5),
    //         // specular_color: Vec4::new(0.7, 0.7, 0.7, 0.1),

    //         // absorption_color: Vec4::new(0.6, 0.1, 0.05, 0.3),
    //         // absorption_color: Vec4::new(0.52, 0.13, 0.1, 0.3),
    //         absorption_color: Vec4::new(0.432, 0.224, 0.133, 0.3),
    //         // absorption_color: Vec4::new(1.0, 209.0 / 255.0, 184.0 / 255.0, 0.8),
    //         // absorption_color: Vec4::new(0.99, 0.521, 0.261, 0.688),
    //         // specular_color: Vec4::new(0.93, 0.48, 0.375, 2.0),
    //         // absorption_color: Vec4::new(0.55, 0.38, 0.07, 0.4),
    //         // specular_color: Vec4::new(0.87, 0.7, 0.38, 0.6),
    //         // specular_color: Vec4::new(0.82, 0.663, 0.365, 0.3),
    //         specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
    //         // specular_color: Vec4::new(0.769, 0.383, 0.325, 0.9),
    //         ambient_factor: 0.15,
    //         ao_factor: 0.1,
    //         eta: 1.55, // index of refraction
    //         // beta: 0.25,   // higher order path roughness
    //         // alpha: 0.05, // first order path roughness
    //         // beta: 0.45,
    //         beta: 0.85,
    //         alpha: 0.35,
    //         shift: 0.1, // specular shift
    //         min_radius_pixels: 0.7,
    //         max_radius_pixels: 2.0,
    //     },
    //     Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
    //     GlobalTransform::default(), // StrandMaterial::default(),
    // ));

    let (squirrel_pose_graph, squirrel_pose_node) = AnimationGraph::from_clip(
        asset_server.load(GltfAssetLabel::Animation(0).from_asset(SQUIRREL_GLTF)),
    );
    commands.insert_resource(SquirrelPoseAnimation {
        graph: animation_graphs.add(squirrel_pose_graph),
        node: squirrel_pose_node,
    });

    commands.spawn((
        Name::new("RedSquirrel_Winter_1 base geometry"),
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(SQUIRREL_GLTF))),
        squirrel_world_transform(),
        GlobalTransform::default(),
    ));
    spawn_squirrel_strand_group(&mut commands, &asset_server);

    commands.spawn((
        FroxelConfig {
            screen_height: 2160,
            screen_width: 2160,
            depth_slices: 16,
            froxel_size_x: 8,
            froxel_size_y: 8,
            ..Default::default()
        },
        Transform::from_xyz(-2.0, 4.5, 0.0).looking_at(Vec3::new(-1.0, 1., 0.), Vec3::Y),
        PanOrbitCamera {
            focus: Vec3::new(0.0, 4.1, 0.0),
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..Default::default()
        },
        Projection::from(PerspectiveProjection {
            fov: 35.0_f32.to_radians(),
            ..default()
        }),
        TieFroxelsToView::Native,
        bevy::core_pipeline::prepass::DepthPrepass,
    ));

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.2))),
        MeshMaterial3d(materials.add(Color::srgb_u8(144, 144, 144))),
        Transform::from_xyz(0.0, 4.5, 0.25),
    ));

    // add one light
    // commands.spawn((
    //     PointLight {
    //         shadows_enabled: true,
    //         ..default()
    //     },
    //     Transform::from_xyz(4.0, 8.0, 4.0),
    // ));

    let default_material = make_default_material(&mut extended_materials);

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(25.0, 25.0))),
        MeshMaterial3d(default_material),
        Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(2.0, 6.0, 0.0),
            rotation: Quat::from_rotation_x(-3.141592 / 4.),
            ..default()
        },
        FroxelConfig {
            // TODO: move into plugin
            screen_width: 8192, // shadow map size in this context
            screen_height: 8192,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
        },
        // The default cascade config is designed to handle large scenes.
        // As this example has a much smaller world, we can tighten the shadow
        // bounds for better visual quality.
        CascadeShadowConfigBuilder {
            // first_cascade_far_bound: 20.0,
            // maximum_distance: 25.0,
            ..default()
        }
        .build(),
    ));
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

/// The egui panel that edits all StrandMaterial components in the world.
fn strand_material_ui_system(
    mut contexts: EguiContexts,
    mut query: Query<&mut StrandMaterial>,
    mut tile_debug: ResMut<TileDebugSettings>,
    mut stochastic_cull: ResMut<StochasticCullSettings>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Strand Material")
        .default_width(300.0)
        .show(ctx, |ui| {
            for mut mat in query.iter_mut() {
                ui.separator();
                ui.label("Tile Occupancy Debug");
                ui.checkbox(&mut tile_debug.enabled, "Enable Heatmap Overlay");
                ui.add(egui::Slider::new(&mut tile_debug.gain, 0.1..=32.0).text("Heatmap Gain"));
                ui.add(egui::Slider::new(&mut tile_debug.alpha, 0.0..=1.0).text("Heatmap Alpha"));
                ui.separator();
                ui.label("Stochastic Culling");
                ui.checkbox(&mut stochastic_cull.enabled, "Enable Stochastic Culling");
                ui.add(
                    egui::Slider::new(&mut stochastic_cull.target_strands_per_pixel, 0.01..=16.0)
                        .logarithmic(true)
                        .text("Target Strands / Pixel"),
                );
                ui.add(
                    egui::Slider::new(&mut stochastic_cull.min_keep_probability, 0.0..=1.0)
                        .text("Min Camera Keep"),
                );
                ui.add(
                    egui::Slider::new(&mut stochastic_cull.shadow_keep_probability, 0.0..=1.0)
                        .text("Shadow Keep"),
                );
                ui.separator();

                // --- Absorption Color ---
                let mut ab = [
                    mat.absorption_color.x,
                    mat.absorption_color.y,
                    mat.absorption_color.z,
                    mat.absorption_color.w,
                ];
                if ui
                    .color_edit_button_rgba_unmultiplied(&mut ab)
                    .on_hover_text("Absorption Color (RGBA)")
                    .changed()
                {
                    mat.absorption_color = Vec4::from(ab);
                }

                // --- Specular Color ---
                let mut sp = [
                    mat.specular_color.x,
                    mat.specular_color.y,
                    mat.specular_color.z,
                    mat.specular_color.w,
                ];
                if ui
                    .color_edit_button_rgba_unmultiplied(&mut sp)
                    .on_hover_text("Specular Color (RGBA)")
                    .changed()
                {
                    mat.specular_color = Vec4::from(sp);
                }

                ui.separator();

                // --- Scalar parameters ---
                ui.add(
                    egui::Slider::new(&mut mat.ambient_factor, 0.0..=1.0).text("Ambient Factor"),
                );
                ui.add(egui::Slider::new(&mut mat.ao_factor, 0.0..=1.0).text("AO Factor"));
                ui.add(
                    egui::Slider::new(&mut mat.eta, 1.0..=2.5).text("Eta (Index of Refraction)"),
                );
                ui.add(egui::Slider::new(&mut mat.beta, 0.0..=1.0).text("Beta (Roughness)"));
                ui.add(egui::Slider::new(&mut mat.alpha, 0.0..=1.0).text("Alpha (Roughness)"));
                ui.add(egui::Slider::new(&mut mat.shift, 0.0..=1.0).text("Specular Shift"));
                ui.add(
                    egui::Slider::new(&mut mat.min_radius_pixels, 0.0..=8.0)
                        .text("Min Strand Radius"),
                );
                ui.add(
                    egui::Slider::new(&mut mat.max_radius_pixels, 0.0..=8.0)
                        .text("Max Strand Radius"),
                );
            }
        });
}

fn rotate_light_around_z_axis(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
) {
    // Angular speed in radians per second (slow rotation)
    const ANGULAR_SPEED: f32 = std::f32::consts::PI / 16.0; // one full rotation in ~32s

    for mut transform in &mut query {
        // Extract position
        let pos = transform.translation;

        // Rotate around global Z by delta_angle
        let delta_angle = ANGULAR_SPEED * time.delta_secs();
        let rotation = Quat::from_rotation_y(delta_angle);
        transform.translation = rotation * pos;

        // Keep the light facing the same *direction* relative to the scene
        // If you want the light to keep pointing toward the origin, uncomment this:
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                backends: Some(Backends::VULKAN),
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | WgpuFeatures::CLEAR_TEXTURE,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(DefaultMaterialPlugin)
        .add_plugins(BevyVsmsPlugin)
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.1)))
        .add_plugins(EguiPlugin::default())
        .add_plugins(StrandRasterizerPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(perf::FpsDisplayPlugin)
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .init_asset::<StrandCacheAsset>()
        .init_asset_loader::<StrandCacheAssetLoader>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (apply_squirrel_pose_animation, rotate_light_around_z_axis),
        )
        .add_systems(EguiPrimaryContextPass, strand_material_ui_system)
        .run();
}

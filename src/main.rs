use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_vsms::prelude::BevyVsmsPlugin;
mod dson;
use dson::*;
mod components;
use components::*;
mod nodes;
mod resources;
use plugin::StrandRasterizerPlugin;
use resources::{StochasticCullSettings, TileDebugSettings};
mod perf;
mod pipelines;
mod plugin;
mod shader_types;
use bevy_gpu_paging_allocator as allocator;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let handle2: Handle<DsonAsset> = asset_server.load("dForce Pixie Cut_708408.dsf".to_string());
    let handle: Handle<DsonAsset> = asset_server.load("curly/curly.dsf".to_string());
    // let handle: Handle<DsonAsset> = asset_server.load("Wet Messy Bang Hair/Wet Messy Bang Hair.dsf".to_string());
    commands.spawn((
        StrandAsset { handle },
        StrandMaterial {
            // absorption_color: Vec4::new(0.7, 0.7, 0.7, 0.5),
            // specular_color: Vec4::new(0.7, 0.7, 0.7, 0.1),

            // absorption_color: Vec4::new(0.6, 0.1, 0.05, 0.3),
            // absorption_color: Vec4::new(0.52, 0.13, 0.1, 0.3),
            // absorption_color: Vec4::new(0.432, 0.224, 0.133, 0.3),
            // absorption_color: Vec4::new(1.0, 209.0 / 255.0, 184.0 / 255.0, 0.8),
            absorption_color: Vec4::new(0.99, 0.521, 0.261, 0.688),
            // specular_color: Vec4::new(0.93, 0.48, 0.375, 2.0),
            // absorption_color: Vec4::new(0.55, 0.38, 0.07, 0.4),
            // specular_color: Vec4::new(0.87, 0.7, 0.38, 0.6),
            // specular_color: Vec4::new(0.82, 0.663, 0.365, 0.3),
            specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
            // specular_color: Vec4::new(0.769, 0.383, 0.325, 0.9),
            ambient_factor: 0.15,
            ao_factor: 0.1,
            eta: 1.55, // index of refraction
            // beta: 0.25,   // higher order path roughness
            // alpha: 0.05, // first order path roughness
            // beta: 0.45,
            beta: 0.85,
            alpha: 0.35,
            shift: 0.1, // specular shift
            pad1: 0,
            pad2: 0,
        },
        // StrandMaterial::default(),
        Transform::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    commands.spawn((
        StrandAsset { handle: handle2 },
        StrandMaterial {
            // absorption_color: Vec4::new(0.7, 0.7, 0.7, 0.5),
            // specular_color: Vec4::new(0.7, 0.7, 0.7, 0.1),

            // absorption_color: Vec4::new(0.6, 0.1, 0.05, 0.3),
            // absorption_color: Vec4::new(0.52, 0.13, 0.1, 0.3),
            absorption_color: Vec4::new(0.432, 0.224, 0.133, 0.3),
            // absorption_color: Vec4::new(1.0, 209.0 / 255.0, 184.0 / 255.0, 0.8),
            // absorption_color: Vec4::new(0.99, 0.521, 0.261, 0.688),
            // specular_color: Vec4::new(0.93, 0.48, 0.375, 2.0),
            // absorption_color: Vec4::new(0.55, 0.38, 0.07, 0.4),
            // specular_color: Vec4::new(0.87, 0.7, 0.38, 0.6),
            // specular_color: Vec4::new(0.82, 0.663, 0.365, 0.3),
            specular_color: Vec4::new(0.99, 0.718, 0.513, 0.422),
            // specular_color: Vec4::new(0.769, 0.383, 0.325, 0.9),
            ambient_factor: 0.15,
            ao_factor: 0.1,
            eta: 1.55, // index of refraction
            // beta: 0.25,   // higher order path roughness
            // alpha: 0.05, // first order path roughness
            // beta: 0.45,
            beta: 0.85,
            alpha: 0.35,
            shift: 0.1, // specular shift
            pad1: 0,
            pad2: 0,
        },
        Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
        GlobalTransform::default(), // StrandMaterial::default(),
    ));

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
            fov: 20.0_f32.to_radians(),
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
        FroxelConfig {
            // TODO: move into plugin
            screen_width: 512, // shadow map size in this context
            screen_height: 512,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
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
                    egui::Slider::new(&mut stochastic_cull.min_dist, 0.0..=300.0)
                        .text("Cull Min Dist"),
                );
                ui.add(
                    egui::Slider::new(&mut stochastic_cull.max_dist, 0.001..=500.0)
                        .text("Cull Max Dist"),
                );
                ui.add(
                    egui::Slider::new(&mut stochastic_cull.exponent, 0.01..=2.0)
                        .text("Cull Exponent"),
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
                    .color_edit_button_rgba_premultiplied(&mut ab)
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
                    .color_edit_button_rgba_premultiplied(&mut sp)
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
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.1)))
        .add_plugins(EguiPlugin::default())
        .add_plugins(StrandRasterizerPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(perf::FpsDisplayPlugin)
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_light_around_z_axis)
        .add_systems(EguiPrimaryContextPass, strand_material_ui_system)
        .run();
}

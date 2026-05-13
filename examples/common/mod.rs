use bevy::{
    light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_vsms::prelude::BevyVsmsPlugin;
use default_material::{DefaultMaterial, DefaultMaterialPlugin, make_default_material};
use strand_software_rasterizer::prelude::*;
use wgpu::Backends;

pub fn add_example_plugins(app: &mut App) -> &mut App {
    app.add_plugins(DefaultPlugins.set(RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings {
            backends: Some(Backends::VULKAN),
            features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | WgpuFeatures::CLEAR_TEXTURE,
            ..Default::default()
        }),
        ..Default::default()
    }))
    .insert_resource(DirectionalLightShadowMap { size: 4096 })
    .add_plugins(DefaultMaterialPlugin)
    .add_plugins(BevyVsmsPlugin)
    .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.1)))
    .add_plugins(EguiPlugin::default())
    .add_plugins(StrandRasterizerPlugin)
    .add_plugins(PanOrbitCameraPlugin)
    .add_systems(EguiPrimaryContextPass, strand_material_ui_system)
}

pub fn spawn_camera(commands: &mut Commands, transform: Transform, focus: Vec3) {
    commands.spawn((
        FroxelConfig {
            screen_height: 2160,
            screen_width: 2160,
            depth_slices: 16,
            froxel_size_x: 8,
            froxel_size_y: 8,
            ..Default::default()
        },
        transform,
        PanOrbitCamera {
            focus,
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
}

pub fn spawn_floor_and_light(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    default_materials: &mut Assets<DefaultMaterial>,
    floor_y: f32,
) {
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.2))),
        MeshMaterial3d(materials.add(Color::srgb_u8(144, 144, 144))),
        Transform::from_xyz(0.0, floor_y + 2.5, 0.25),
    ));

    let default_material = make_default_material(default_materials);
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(25.0, 25.0))),
        MeshMaterial3d(default_material),
        Transform::from_translation(Vec3::new(0.0, floor_y, 0.0)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(2.0, floor_y + 4.0, 0.0),
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4),
            ..default()
        },
        FroxelConfig {
            screen_width: 8192,
            screen_height: 8192,
            froxel_size_x: 8,
            froxel_size_y: 8,
            depth_slices: 16,
        },
        CascadeShadowConfigBuilder::default().build(),
    ));
}

pub fn rotate_light_around_z_axis(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
) {
    const ANGULAR_SPEED: f32 = std::f32::consts::PI / 16.0;

    for mut transform in &mut query {
        let delta_angle = ANGULAR_SPEED * time.delta_secs();
        transform.translation = Quat::from_rotation_y(delta_angle) * transform.translation;
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

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

                let mut absorption = mat.absorption_color.to_array();
                if ui
                    .color_edit_button_rgba_unmultiplied(&mut absorption)
                    .on_hover_text("Absorption Color (RGBA)")
                    .changed()
                {
                    mat.absorption_color = Vec4::from(absorption);
                }

                let mut specular = mat.specular_color.to_array();
                if ui
                    .color_edit_button_rgba_unmultiplied(&mut specular)
                    .on_hover_text("Specular Color (RGBA)")
                    .changed()
                {
                    mat.specular_color = Vec4::from(specular);
                }

                ui.separator();
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

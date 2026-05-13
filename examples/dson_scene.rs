mod common;

use bevy::{asset::AssetApp, prelude::*};
use bevy_dson::{DsonAsset, DsonAssetLoader};
use default_material::DefaultMaterial;
use serde_json::Value;
use strand_software_rasterizer::prelude::*;

#[derive(Component)]
struct DsonStrandSource {
    handle: Handle<DsonAsset>,
    asset_path: &'static str,
}

fn load_dson_polyline_strands(asset_path: &str) -> Option<Vec<Vec<u32>>> {
    let path = std::path::Path::new("assets").join(asset_path);
    let json: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let geometries = json.get("geometry_library")?.as_array()?;

    for geometry in geometries {
        let Some(values) = geometry
            .get("polyline_list")
            .and_then(|polyline_list| polyline_list.get("values"))
            .and_then(Value::as_array)
        else {
            continue;
        };

        let strands = values
            .iter()
            .filter_map(Value::as_array)
            .filter(|strand| strand.len() >= 4)
            .map(|strand| {
                strand[2..]
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|index| index as u32)
                    .collect::<Vec<_>>()
            })
            .filter(|strand| strand.len() >= 2)
            .collect::<Vec<_>>();

        if !strands.is_empty() {
            return Some(strands);
        }
    }

    None
}

fn pack_loaded_dson_strands(
    mut commands: Commands,
    sources: Query<(Entity, &DsonStrandSource), Without<StrandList>>,
    dson_assets: Res<Assets<DsonAsset>>,
) {
    for (entity, source) in &sources {
        let Some(asset) = dson_assets.get(&source.handle) else {
            continue;
        };
        let Some((_, geometry)) = asset
            .dson_file
            .geometry_library
            .as_ref()
            .and_then(|library| library.values().next())
        else {
            warn!("Geometry library not found for {}", source.asset_path);
            commands.entity(entity).remove::<DsonStrandSource>();
            continue;
        };
        let Some(strands) = load_dson_polyline_strands(source.asset_path) else {
            warn!("Polyline list not found for {}", source.asset_path);
            commands.entity(entity).remove::<DsonStrandSource>();
            continue;
        };

        let vertices = geometry
            .vertices
            .values
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]) * 0.0254)
            .collect();

        commands
            .entity(entity)
            .insert(StrandList { vertices, strands })
            .remove::<DsonStrandSource>();
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut default_materials: ResMut<Assets<DefaultMaterial>>,
) {
    let curly_hair = "curly/curly.dsf";
    commands.spawn((
        DsonStrandSource {
            handle: asset_server.load(curly_hair),
            asset_path: curly_hair,
        },
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

    let pixie_cut = "dForce Pixie Cut_708408.dsf";
    commands.spawn((
        DsonStrandSource {
            handle: asset_server.load(pixie_cut),
            asset_path: pixie_cut,
        },
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
        .init_asset::<DsonAsset>()
        .init_asset_loader::<DsonAssetLoader>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (pack_loaded_dson_strands, common::rotate_light_around_z_axis),
        )
        .run();
}

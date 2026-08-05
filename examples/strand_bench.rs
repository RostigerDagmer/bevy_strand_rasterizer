//! Repeatable GPU micro-benchmark for the strand prepass and software rasterizer.
//!
//! Examples:
//!   cargo run --release --example strand_bench -- --case squirrel --headless
//!   cargo run --release --example strand_bench -- --case clustered-grid --display
//!   cargo run --release --example strand_bench -- --help

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    diagnostic::DiagnosticsStore,
    light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    log::LogPlugin,
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::TextureFormat,
        renderer::RenderAdapterInfo,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    window::{ExitCondition, PresentMode, WindowCloseRequested, WindowResolution},
    winit::WinitPlugin,
};
use bevy_gpu_paging_allocator::GpuPagingAllocatorPlugin;
use bevy_vsms::prelude::BevyVsmsPlugin;
use serde_json::{Map, Value, json};
use strand_software_rasterizer::prelude::*;
use wgpu::Backends;

const SQUIRREL_SPLIT_DIR: &str = "assets/squirrel/RedSquirrel_Winter_1.strands.d";
const SQUIRREL_SPLIT_ASSET_DIR: &str = "squirrel/RedSquirrel_Winter_1.strands.d";
const SQUIRREL_MONOLITHIC: &str = "assets/squirrel/RedSquirrel_Winter_1.strands";
const SQUIRREL_MONOLITHIC_ASSET: &str = "squirrel/RedSquirrel_Winter_1.strands";

#[derive(Clone, Resource)]
struct BenchConfig {
    case: String,
    headless: bool,
    exit_after_bench: bool,
    width: u32,
    height: u32,
    warmup_frames: usize,
    sample_frames: usize,
    telemetry: bool,
    fine_binning: FineBinningBackend,
    output: PathBuf,
    asset: Option<String>,
    clusters: u32,
    strands_per_cluster: u32,
    points_per_strand: u32,
    seed: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            case: "squirrel-body".into(),
            headless: true,
            exit_after_bench: true,
            width: 1920,
            height: 1080,
            warmup_frames: 120,
            sample_frames: 300,
            telemetry: true,
            fine_binning: FineBinningBackend::SegmentScatter,
            output: PathBuf::from("strand-bench.json"),
            asset: None,
            clusters: 125,
            strands_per_cluster: 128,
            points_per_strand: 16,
            seed: 0x5eed_cafe,
        }
    }
}

impl BenchConfig {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("missing value after {arg}"))
            };
            match arg.as_str() {
                "--headless" => {
                    config.headless = true;
                    config.exit_after_bench = true;
                }
                "--display" => {
                    config.headless = false;
                    config.exit_after_bench = false;
                }
                "--exit-after-bench" => config.exit_after_bench = true,
                "--case" => config.case = value()?,
                "--asset" => config.asset = Some(value()?),
                "--width" => config.width = parse_u32(&arg, &value()?)?,
                "--height" => config.height = parse_u32(&arg, &value()?)?,
                "--warmup" => config.warmup_frames = parse_usize(&arg, &value()?)?,
                "--samples" => config.sample_frames = parse_usize(&arg, &value()?)?,
                "--no-telemetry" => config.telemetry = false,
                "--fine-binning" => {
                    config.fine_binning = match value()?.as_str() {
                        "segment" => FineBinningBackend::SegmentScatter,
                        "page-csr" => FineBinningBackend::PageCsr,
                        value => {
                            return Err(format!(
                                "invalid --fine-binning backend: {value} (expected segment or page-csr)"
                            ));
                        }
                    }
                }
                "--output" => config.output = PathBuf::from(value()?),
                "--clusters" => config.clusters = parse_u32(&arg, &value()?)?,
                "--strands-per-cluster" => config.strands_per_cluster = parse_u32(&arg, &value()?)?,
                "--points-per-strand" => {
                    config.points_per_strand = parse_u32(&arg, &value()?)?.max(2)
                }
                "--seed" => config.seed = parse_u64(&arg, &value()?)?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        if !matches!(
            config.case.as_str(),
            "squirrel" | "squirrel-body" | "asset" | "clustered-grid" | "length-skew" | "hotspot"
        ) {
            return Err(format!("unknown benchmark case: {}", config.case));
        }
        if config.case == "asset" && config.asset.is_none() {
            return Err("--case asset requires --asset <path relative to assets/>".into());
        }
        if config.sample_frames == 0 {
            return Err("--samples must be greater than zero".into());
        }
        Ok(config)
    }
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn print_help() {
    println!(
        "strand_bench [OPTIONS]\n\
         \nCases:\n\
           squirrel          all split squirrel strand groups (or monolithic fallback)\n\
           squirrel-body     the large Body01 real-asset group (default)\n\
           asset             one cache supplied with --asset, relative to assets/\n\
           clustered-grid    many small entity-local clusters in a uniform 3D grid\n\
           length-skew       heavy-tailed strand lengths in one entity\n\
           hotspot           dense, screen-overlapping strands\n\
         \nOptions:\n\
           --headless | --display       offscreen automation or an inspectable window\n\
           --exit-after-bench           also close display mode after writing results\n\
           --case CASE                  workload preset\n\
           --asset PATH                 cache path relative to assets/ for case asset\n\
           --width N --height N         render resolution (default 1920x1080)\n\
           --warmup N --samples N       warmup and measured GPU frames\n\
           --no-telemetry               disable prepass counters/readbacks\n\
           --fine-binning BACKEND        segment (default) or page-csr\n\
           --output PATH                JSON result path (default strand-bench.json)\n\
           --clusters N                 clustered-grid entity count (default 125)\n\
           --strands-per-cluster N      synthetic strand count (default 128)\n\
           --points-per-strand N        synthetic base point count (default 16)\n\
           --seed N                     deterministic synthetic seed"
    );
}

#[derive(Resource)]
struct BenchState {
    started: std::time::Instant,
    expected_geometries: usize,
    warmup_seen: usize,
    measured: usize,
    last_raster_sample: Option<bevy::platform::time::Instant>,
    samples: BTreeMap<String, Vec<f64>>,
    finished: bool,
    shutdown_frames_remaining: Option<u32>,
}

#[derive(Resource, Default, Clone)]
struct AdapterMetadata {
    name: String,
    backend: String,
    driver: String,
    driver_info: String,
}

struct AdapterMetadataPlugin;

impl Plugin for AdapterMetadataPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let metadata = app
            .get_sub_app(RenderApp)
            .and_then(|render_app| render_app.world().get_resource::<RenderAdapterInfo>())
            .map(|info| AdapterMetadata {
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                driver: info.driver.clone(),
                driver_info: info.driver_info.clone(),
            })
            .unwrap_or_default();
        app.insert_resource(metadata);
    }
}

fn main() {
    let config = BenchConfig::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}\n\nRun with --help for usage.");
        std::process::exit(2);
    });
    let telemetry_enabled = config.telemetry;
    let fine_binning = config.fine_binning;
    let mut app = App::new();

    let render_plugin = RenderPlugin {
        render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
            backends: Some(Backends::VULKAN),
            features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | WgpuFeatures::CLEAR_TEXTURE
                | WgpuFeatures::TIMESTAMP_QUERY
                | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_PASSES
                | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            ..default()
        })),
        ..default()
    };

    if config.headless {
        app.add_plugins(
            DefaultPlugins
                .set(render_plugin)
                .set(AssetPlugin {
                    file_path: format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
                    ..default()
                })
                .set(LogPlugin {
                    filter: "wgpu=warn,bevy_render=info,strand_bench=info,strand_software_rasterizer=error".into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(render_plugin)
                .set(AssetPlugin {
                    file_path: format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
                    ..default()
                })
                .set(LogPlugin {
                    filter: "wgpu=warn,bevy_render=info,strand_bench=info,strand_software_rasterizer=error".into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("Strand benchmark: {}", config.case),
                        resolution: WindowResolution::new(config.width, config.height),
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                }),
        );
    }

    app.insert_resource(config)
        .insert_resource(PrepassTelemetrySettings {
            enabled: telemetry_enabled,
        })
        .insert_resource(fine_binning)
        .insert_resource(DirectionalLightShadowMap { size: 2048 })
        .insert_resource(ClearColor(Color::srgb(0.025, 0.025, 0.03)))
        .add_plugins((
            GpuPagingAllocatorPlugin,
            BevyVsmsPlugin,
            StrandRasterizerPlugin,
            RenderDiagnosticsPlugin,
            AdapterMetadataPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (collect_benchmark_samples, hard_exit_display_on_close),
        );
    app.run();
}

fn setup(
    mut commands: Commands,
    config: Res<BenchConfig>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
) {
    let expected_geometries = match config.case.as_str() {
        "squirrel" => spawn_squirrel(&mut commands, &asset_server, false),
        "squirrel-body" => spawn_squirrel(&mut commands, &asset_server, true),
        "asset" => {
            let path = config.asset.as_ref().unwrap();
            commands.spawn((
                Name::new(format!("benchmark asset: {path}")),
                StrandCache {
                    handle: asset_server.load(path.clone()),
                },
                benchmark_material(),
                Transform::default(),
                GlobalTransform::default(),
            ));
            1
        }
        "clustered-grid" => spawn_clustered_grid(&mut commands, &config),
        "length-skew" => spawn_length_skew(&mut commands, &config),
        "hotspot" => spawn_hotspot(&mut commands, &config),
        _ => unreachable!(),
    };

    let camera_transform = match config.case.as_str() {
        "squirrel" | "squirrel-body" => {
            Transform::from_xyz(-3.0, 1.0, 0.0).looking_at(Vec3::new(0.0, 2.5, 0.0), Vec3::Y)
        }
        _ => Transform::from_xyz(7.5, 5.5, 9.5).looking_at(Vec3::ZERO, Vec3::Y),
    };
    let camera_target = if config.headless {
        let image = Image::new_target_texture(
            config.width,
            config.height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        );
        RenderTarget::Image(images.add(image).into())
    } else {
        RenderTarget::default()
    };

    commands.spawn((
        Name::new("benchmark camera"),
        Camera3d::default(),
        Camera::default(),
        camera_target,
        Projection::from(PerspectiveProjection {
            fov: 35.0_f32.to_radians(),
            ..default()
        }),
        camera_transform,
        FroxelConfig {
            screen_width: config.width,
            screen_height: config.height,
            depth_slices: 16,
            froxel_size_x: 8,
            froxel_size_y: 8,
        },
        TieFroxelsToView::Native,
        bevy::core_pipeline::prepass::DepthPrepass,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.6, 0.0)),
        FroxelConfig {
            screen_width: 2048,
            screen_height: 2048,
            depth_slices: 16,
            froxel_size_x: 8,
            froxel_size_y: 8,
        },
        CascadeShadowConfigBuilder::default().build(),
    ));
    commands.insert_resource(BenchState {
        started: std::time::Instant::now(),
        expected_geometries,
        warmup_seen: 0,
        measured: 0,
        last_raster_sample: None,
        samples: BTreeMap::new(),
        finished: false,
        shutdown_frames_remaining: None,
    });
    info!(
        "benchmark '{}' loading {} strand geometries; {} warmup + {} measured frames",
        config.case, expected_geometries, config.warmup_frames, config.sample_frames
    );
}

fn spawn_squirrel(commands: &mut Commands, asset_server: &AssetServer, body_only: bool) -> usize {
    let transform =
        Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)).with_scale(Vec3::splat(8.0));
    if body_only {
        let file = "RedSquirrel_Winter_Hair_Body01.strands";
        commands.spawn((
            Name::new(file),
            StrandCache {
                handle: asset_server.load(format!("{SQUIRREL_SPLIT_ASSET_DIR}/{file}")),
            },
            benchmark_material(),
            transform,
            GlobalTransform::default(),
        ));
        return 1;
    }

    let mut files = fs::read_dir(SQUIRREL_SPLIT_DIR)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "strands"))
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() && Path::new(SQUIRREL_MONOLITHIC).exists() {
        commands.spawn((
            Name::new("squirrel monolithic strands"),
            StrandCache {
                handle: asset_server.load(SQUIRREL_MONOLITHIC_ASSET),
            },
            benchmark_material(),
            transform,
            GlobalTransform::default(),
        ));
        return 1;
    }
    assert!(
        !files.is_empty(),
        "squirrel strand assets were not found under {SQUIRREL_SPLIT_DIR}"
    );
    let count = files.len();
    commands
        .spawn((
            Name::new("squirrel strand groups"),
            transform,
            GlobalTransform::default(),
        ))
        .with_children(|parent| {
            for path in files {
                let file = path.file_name().unwrap().to_string_lossy().into_owned();
                parent.spawn((
                    Name::new(file.clone()),
                    StrandCache {
                        handle: asset_server.load(format!("{SQUIRREL_SPLIT_ASSET_DIR}/{file}")),
                    },
                    benchmark_material(),
                    Transform::default(),
                    GlobalTransform::default(),
                ));
            }
        });
    count
}

fn benchmark_material() -> StrandMaterial {
    StrandMaterial {
        absorption_color: Vec4::new(0.56, 0.25, 0.14, 0.45),
        specular_color: Vec4::new(0.9, 0.72, 0.52, 0.2),
        ambient_factor: 0.15,
        ao_factor: 0.1,
        eta: 1.55,
        beta: 0.85,
        alpha: 0.35,
        shift: 0.1,
        min_radius_pixels: 0.3,
        max_radius_pixels: 1.4,
    }
}

fn spawn_clustered_grid(commands: &mut Commands, config: &BenchConfig) -> usize {
    let side = (config.clusters as f32).cbrt().ceil() as u32;
    for cluster in 0..config.clusters {
        let x = cluster % side;
        let y = (cluster / side) % side;
        let z = cluster / (side * side);
        let denominator = (side.saturating_sub(1)).max(1) as f32;
        let position = Vec3::new(
            -3.5 + 7.0 * x as f32 / denominator,
            -2.5 + 5.0 * y as f32 / denominator,
            -3.5 + 7.0 * z as f32 / denominator,
        );
        commands.spawn((
            Name::new(format!("synthetic cluster {cluster}")),
            make_cluster(
                config.seed.wrapping_add(cluster as u64),
                config.strands_per_cluster,
                config.points_per_strand,
                0.24,
                0.7,
            ),
            benchmark_material(),
            Transform::from_translation(position),
            GlobalTransform::default(),
        ));
    }
    config.clusters as usize
}

fn spawn_length_skew(commands: &mut Commands, config: &BenchConfig) -> usize {
    let count = config.clusters.saturating_mul(config.strands_per_cluster);
    let mut rng = Lcg::new(config.seed);
    let mut vertices = Vec::new();
    let mut strands = Vec::with_capacity(count as usize);
    for strand_index in 0..count {
        let bucket = rng.next_u32() % 100;
        let points = if bucket < 70 {
            4
        } else if bucket < 93 {
            config.points_per_strand
        } else {
            config.points_per_strand.saturating_mul(4)
        }
        .max(2);
        let root = Vec3::new(rng.signed() * 3.5, rng.signed() * 2.5, rng.signed() * 3.5);
        strands.push(append_strand(
            &mut vertices,
            root,
            points,
            0.015,
            0.8 + rng.unit() * 1.8,
            strand_index,
        ));
    }
    commands.spawn((
        Name::new("synthetic heavy-tailed lengths"),
        StrandList { vertices, strands },
        benchmark_material(),
        Transform::default(),
        GlobalTransform::default(),
    ));
    1
}

fn spawn_hotspot(commands: &mut Commands, config: &BenchConfig) -> usize {
    let strands = config.clusters.saturating_mul(config.strands_per_cluster);
    commands.spawn((
        Name::new("synthetic projection hotspot"),
        make_cluster(config.seed, strands, config.points_per_strand, 0.08, 4.0),
        benchmark_material(),
        Transform::default(),
        GlobalTransform::default(),
    ));
    1
}

fn make_cluster(seed: u64, strand_count: u32, points: u32, radius: f32, length: f32) -> StrandList {
    let mut rng = Lcg::new(seed);
    let mut vertices = Vec::with_capacity(strand_count.saturating_mul(points) as usize);
    let mut strands = Vec::with_capacity(strand_count as usize);
    for strand in 0..strand_count {
        let root = Vec3::new(
            rng.signed() * radius,
            rng.signed() * radius,
            rng.signed() * radius,
        );
        strands.push(append_strand(
            &mut vertices,
            root,
            points,
            radius * 0.08,
            length * (0.7 + 0.6 * rng.unit()),
            strand,
        ));
    }
    StrandList { vertices, strands }
}

fn append_strand(
    vertices: &mut Vec<Vec3>,
    root: Vec3,
    points: u32,
    curl: f32,
    length: f32,
    strand: u32,
) -> Vec<u32> {
    let mut indices = Vec::with_capacity(points as usize);
    let phase = strand as f32 * 2.399_963_1;
    for point in 0..points {
        let t = point as f32 / (points - 1).max(1) as f32;
        indices.push(vertices.len() as u32);
        vertices.push(
            root + Vec3::new(
                (phase + t * 8.0).cos() * curl * t,
                length * t,
                (phase + t * 8.0).sin() * curl * t,
            ),
        );
    }
    indices
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

fn collect_benchmark_samples(
    config: Res<BenchConfig>,
    diagnostics: Res<DiagnosticsStore>,
    adapter: Res<AdapterMetadata>,
    geometries: Query<&StrandGeometry>,
    mut state: ResMut<BenchState>,
    mut exit: MessageWriter<AppExit>,
) {
    if state.finished {
        if let Some(frames) = state.shutdown_frames_remaining {
            if frames == 0 {
                exit.write(AppExit::Success);
            } else {
                state.shutdown_frames_remaining = Some(frames - 1);
            }
        }
        return;
    }
    if !state.finished && config.headless && state.started.elapsed() > Duration::from_secs(120) {
        error!(
            "benchmark timed out waiting for render pipelines and GPU timestamps; check earlier shader or asset errors"
        );
        state.finished = true;
        exit.write(AppExit::error());
        return;
    }
    if geometries.iter().count() < state.expected_geometries {
        return;
    }
    let raster_path =
        bevy::diagnostic::DiagnosticPath::new("render/strand_rasterizer/rasterize/elapsed_gpu");
    let Some(raster_measurement) = diagnostics.get_measurement(&raster_path) else {
        return;
    };
    let shading_path =
        bevy::diagnostic::DiagnosticPath::new("render/strand_shading/shade/elapsed_gpu");
    if diagnostics.get_measurement(&shading_path).is_none() {
        return;
    }
    if config.telemetry {
        let telemetry_path = bevy::diagnostic::DiagnosticPath::new(
            "render/strand_prepass/telemetry/allocated_pages",
        );
        if diagnostics.get_measurement(&telemetry_path).is_none() {
            return;
        }
    }
    if state.last_raster_sample == Some(raster_measurement.time) {
        return;
    }
    state.last_raster_sample = Some(raster_measurement.time);
    if state.warmup_seen < config.warmup_frames {
        state.warmup_seen += 1;
        if state.warmup_seen == config.warmup_frames {
            info!(
                "warmup complete; collecting {} frames",
                config.sample_frames
            );
        }
        return;
    }

    let mut telemetry_histogram = [0_u64; 32];
    let mut page_candidates = None;
    let mut active_candidate_pages = None;
    let mut frustum_retention = BTreeMap::<usize, (Option<f64>, Option<f64>)>::new();
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        if (path.starts_with("render/strand_prepass/")
            || path.starts_with("render/strand_rasterizer/")
            || path.starts_with("render/strand_shadow_rasterizer/")
            || path.starts_with("render/strand_shading/"))
            && path.ends_with("/elapsed_gpu")
            && let Some(value) = diagnostic.value()
        {
            state
                .samples
                .entry(path.to_owned())
                .or_default()
                .push(value);
        } else if path.starts_with("render/strand_prepass/telemetry/")
            && let Some(value) = diagnostic.value()
        {
            state
                .samples
                .entry(path.to_owned())
                .or_default()
                .push(value);
            if let Some(bin) = path
                .strip_prefix("render/strand_prepass/telemetry/candidate_histogram/")
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&bin| bin < telemetry_histogram.len())
            {
                telemetry_histogram[bin] = value as u64;
            } else if path.ends_with("/page_candidates") {
                page_candidates = Some(value);
            } else if path.ends_with("/active_candidate_pages") {
                active_candidate_pages = Some(value);
            } else if let Some(rest) = path.strip_prefix("render/strand_prepass/telemetry/frustum/")
                && let Some((frustum_id, field)) = rest.split_once('/')
                && let Ok(frustum_id) = frustum_id.parse::<usize>()
            {
                let entry = frustum_retention.entry(frustum_id).or_default();
                match field {
                    "visible_strands" => entry.0 = Some(value),
                    "retained_strands" => entry.1 = Some(value),
                    _ => {}
                }
            }
        }
    }
    if let (Some(candidates), Some(pages)) = (page_candidates, active_candidate_pages)
        && pages > 0.0
    {
        state
            .samples
            .entry("derived/candidates_per_active_page/mean".into())
            .or_default()
            .push(candidates / pages);
    }
    for (name, fraction) in [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)] {
        if let Some(upper_bound) = histogram_percentile_upper(&telemetry_histogram, fraction) {
            state
                .samples
                .entry(format!(
                    "derived/candidates_per_active_page/{name}_upper_bound"
                ))
                .or_default()
                .push(upper_bound as f64);
        }
    }
    for (frustum_id, (visible, retained)) in frustum_retention {
        if let (Some(visible), Some(retained)) = (visible, retained)
            && visible > 0.0
        {
            state
                .samples
                .entry(format!("derived/frustum/{frustum_id}/retention_ratio"))
                .or_default()
                .push(retained / visible);
        }
    }
    state.measured += 1;
    if state.measured < config.sample_frames {
        return;
    }

    let (strand_count, vertex_count) = geometries.iter().fold((0_u64, 0_u64), |acc, geometry| {
        (
            acc.0 + u64::from(geometry.strand_count),
            acc.1 + u64::from(geometry.index_count),
        )
    });
    let report = make_report(
        &config,
        &adapter,
        &state.samples,
        state.expected_geometries,
        strand_count,
        vertex_count,
    );
    let encoded = serde_json::to_string_pretty(&report).expect("benchmark report must serialize");
    if let Some(parent) = config
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    fs::write(&config.output, &encoded)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", config.output.display()));
    println!("{encoded}");
    info!("benchmark written to {}", config.output.display());
    state.finished = true;
    if config.exit_after_bench {
        if config.headless {
            // Let render diagnostics finish mapping outstanding query buffers before teardown.
            state.shutdown_frames_remaining = Some(8);
        } else {
            // Winit + this renderer currently faults while destroying the Vulkan device. The
            // report is synchronously persisted, so skip that broken teardown path.
            hard_process_exit(0);
        }
    } else {
        info!("display remains open for inspection; close the window to exit");
    }
}

fn hard_exit_display_on_close(
    config: Res<BenchConfig>,
    mut close_requests: MessageReader<WindowCloseRequested>,
) {
    if !config.headless && close_requests.read().next().is_some() {
        hard_process_exit(0);
    }
}

fn hard_process_exit(code: i32) -> ! {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn _exit(status: i32) -> !;
        }
        // SAFETY: the benchmark report is synchronously written before this path is used. `_exit`
        // deliberately skips the faulty Vulkan/Winit atexit teardown and cannot return.
        unsafe { _exit(code) }
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

fn make_report(
    config: &BenchConfig,
    adapter: &AdapterMetadata,
    samples: &BTreeMap<String, Vec<f64>>,
    geometry_count: usize,
    strand_count: u64,
    vertex_count: u64,
) -> Value {
    let mut timings = Map::new();
    let mut telemetry = Map::new();
    for (name, values) in samples {
        if name.ends_with("/elapsed_gpu") {
            timings.insert(name.clone(), summarize(values));
        } else if let Some(name) = name.strip_prefix("render/strand_prepass/telemetry/") {
            telemetry.insert(name.to_owned(), summarize(values));
        } else if let Some(name) = name.strip_prefix("derived/") {
            telemetry.insert(name.to_owned(), summarize(values));
        }
    }
    json!({
        "schema_version": 1,
        "case": config.case,
        "mode": if config.headless { "headless" } else { "display" },
        "resolution": { "width": config.width, "height": config.height },
        "warmup_frames": config.warmup_frames,
        "sample_frames": config.sample_frames,
        "telemetry_enabled": config.telemetry,
        "fine_binning_backend": match config.fine_binning {
            FineBinningBackend::SegmentScatter => "segment",
            FineBinningBackend::PageCsr => "page-csr",
        },
        "seed": config.seed,
        "workload": {
            "geometry_count": geometry_count,
            "strand_count": strand_count,
            "vertex_count": vertex_count,
            "clusters": config.clusters,
            "strands_per_cluster": config.strands_per_cluster,
            "points_per_strand": config.points_per_strand,
            "asset": config.asset,
        },
        "adapter": {
            "name": adapter.name,
            "backend": adapter.backend,
            "driver": adapter.driver,
            "driver_info": adapter.driver_info,
        },
        "timings_ms": timings,
        "telemetry": telemetry,
    })
}

fn histogram_percentile_upper(histogram: &[u64; 32], fraction: f64) -> Option<u64> {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return None;
    }
    let target = ((total as f64 * fraction).ceil() as u64).max(1);
    let mut cumulative = 0_u64;
    for (bin, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return Some(if bin == 31 {
                u32::MAX as u64
            } else {
                (1_u64 << (bin + 1)) - 1
            });
        }
    }
    None
}

fn summarize(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let percentile =
        |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction).round() as usize];
    json!({
        "samples": sorted.len(),
        "min": sorted[0],
        "mean": mean,
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": sorted[sorted.len() - 1],
    })
}

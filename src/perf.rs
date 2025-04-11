use bevy::{
    app::{Plugin, Startup, Update},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    text::{TextColor, TextSpan},
    ui::{BackgroundColor, Node, PositionType, UiRect, Val},
};

#[derive(Component)]
struct FpsValueMarker;

pub struct FpsDisplayPlugin;

impl Plugin for FpsDisplayPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GlobalZIndex>();

        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, setup_fps_ui_strict)
            .add_systems(Update, update_fps_value);
    }
}

fn setup_fps_ui_strict(mut commands: Commands) {
    let font = TextFont {
        font_size: 10.0,
        ..Default::default()
    };

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                padding: UiRect::all(Val::Px(5.0)),
                ..default()
            },
            BackgroundColor(Color::BLACK.with_alpha(0.75)),
            GlobalZIndex(i32::MAX),
        ))
        .with_children(|parent| {
            parent
                .spawn((Text::default(), FpsValueMarker))
                .with_children(|text_node_children| {
                    text_node_children.spawn((
                        TextSpan::new("FPS: "),
                        TextColor::WHITE,
                        font.clone(),
                    ));

                    text_node_children.spawn((
                        TextSpan::new("..."),
                        TextColor::WHITE,
                        font.clone(),
                    ));
                });
        });
}

fn update_fps_value(
    diagnostics: Res<DiagnosticsStore>,
    query: Query<Entity, With<FpsValueMarker>>,
    mut writer: TextUiWriter,
) {
    if let Ok(entity) = query.get_single() {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                *writer.text(entity, 2) = format!("{value:.1}").into();
            }
        }
    }
}

// --- App Setup ---

// fn main() {
//     App::new()
//         .add_plugins(DefaultPlugins)
//         .add_plugins(FpsDisplayPlugin)
//         .run();
// }

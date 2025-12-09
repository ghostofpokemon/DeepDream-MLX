# pylint: disable=too-many-lines
"""A Textual user interface for DeepDream."""

import json
import subprocess
from datetime import datetime
from typing import Optional
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    RadioButton,
    RadioSet,
    Static,
)

from config import ARG_MAPPING, PRESETS
from widgets import FilteredDirectoryTree, ImagePreview
from deepdream import list_models


class Tui(App):
    """A Textual user interface for DeepDream."""

    CSS_PATH = "tui.css"

    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle dark mode"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        """Initialize the TUI."""
        super().__init__()
        self.selected_file = None
        self.current_preset_name = None
        self.history_data = {}
        self.latest_output_path = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(
            self._compose_left_pane(),
            self._compose_center_pane(),
            self._compose_right_pane(),
            id="main-content-area",
        )
        yield Log(id="log")
        yield Footer()

    def _compose_left_pane(self) -> Vertical:
        """Compose the left pane of the TUI."""
        return Vertical(
            Static("Input Image", classes="header-text", id="input-header"),
            FilteredDirectoryTree("./input", id="tree-view"),
            Static("Outputs", classes="header-text", id="output-header"),
            FilteredDirectoryTree("./outputs", id="output-tree"),
            Static("History", classes="header-text", id="history-header"),
            DataTable(id="history-table"),
            id="left-pane",
        )

    def _compose_center_pane(self) -> Vertical:
        """Compose the center pane of the TUI."""
        return Vertical(
            Horizontal(
                Vertical(
                    Static("Input Preview", classes="header-text", id="input-preview-header"),
                    ImagePreview(id="input-preview"),
                    id="input-preview-container",
                ),
                Vertical(
                    Static("Output Preview", classes="header-text", id="output-preview-header"),
                    ImagePreview(id="output-preview"),
                    id="output-preview-container",
                ),
                id="preview-row",
            ),
            Static(
                "Input: — | Preset: — | Model: — | Output: —",
                id="status-strip",
            ),
            id="center-pane",
        )

    def _compose_right_pane(self) -> Vertical:
        """Compose the right pane of the TUI."""
        return Vertical(
            Horizontal(
                Vertical(
                    Static("Presets", classes="header-text", id="presets-header"),
                    *[
                        Button(preset["name"], id=preset["name"], variant="default")
                        for preset in PRESETS
                    ],
                ),
                Vertical(
                    Static("Model", classes="header-text", id="model-header"),
                    RadioSet(
                        *[RadioButton(m, value=(m == "vgg16")) for m in list_models()],
                        id="model-selection",
                    ),
                ),
            ),
            Horizontal(
                Static(
                    "Select a preset or a past run",
                    id="preset-details-placeholder",
                ),
                id="preset-details-container",
            ),
            id="right-pane",
        )

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        table = self.query_one(DataTable)
        table.add_columns("Preset", "Model", "Date")
        Path("outputs").mkdir(exist_ok=True)
        # Default to the first preset so the detail panel is ready.
        self.current_preset_name = PRESETS[0]["name"]
        self.populate_preset_details(PRESETS[0])
        self._update_status_strip()
        self._set_preview_placeholder("#input-preview", "Select an input to preview")
        self._set_preview_placeholder("#output-preview", "No output yet")
        
        # Futurist Banner (ASCII)
        banner = [
            "[bold cyan]╔══════════════════════════════════════════════════════════════════════╗[/]",
            "[bold cyan]║                                                                      ║[/]",
            "[bold cyan]║              [bold pink]D E E P   D R E A M   M L X[/]                       ║[/]",
            "[bold cyan]║         [dim blue]FUTURIST • BEAUX-ARTS • ARCHITECTURE[/]                 ║[/]",
            "[bold cyan]║                                                                      ║[/]",
            "[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/]"
        ]
        log = self.query_one("#log")
        for line in banner:
            log.write_line(line)
        log.write_line("[bold purple]System Ready. Pick an input, choose a preset, and DREAM.[/]")

    def on_ready(self) -> None:
        """Called when the app is ready (layout complete)."""
        if not self.selected_file:
            self._auto_select_first_input()

    def on_directory_tree_file_selected(
        self, event: "DirectoryTree.FileSelected"
    ) -> None:
        """Called when the user clicks a file in the directory tree."""
        event.stop()
        tree_id = getattr(event, "control", None).id if hasattr(event, "control") else ""
        if tree_id == "output-tree":
            self._show_output_preview(event.path)
            self._update_status_strip(event.path)
            return

        self.selected_file = event.path
        self.query_one("#log").write_line(f"Selected file: {self.selected_file}")
        self._show_input_preview(event.path)
        self.update_history_table()
        self._show_latest_output_for_input()
        self._update_status_strip()

    def update_history_table(self):
        """Scan for output files and update the history table."""
        table = self.query_one(DataTable)
        table.clear()
        self.history_data = {}

        if not self.selected_file:
            return

        input_stem = Path(self.selected_file).stem
        glob_pattern = f"outputs/{input_stem}_tui_*.json"
        json_files = sorted(
            Path(".").glob(glob_pattern),
            key=lambda p: p.stem,
            reverse=True,
        )
        for json_file_path in json_files:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                self.history_data[json_file_path.stem] = data
                table.add_row(
                    data["preset_name"],
                    data["model"],
                    data["timestamp"],
                    key=json_file_path.stem,
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button is pressed."""
        event.stop()
        if event.button.id == "run-dream":
            self.run_dream()
            return

        preset_name = event.button.id
        preset = next((preset for preset in PRESETS if preset["name"] == preset_name), None)

        if preset:
            self.current_preset_name = preset_name
            self.populate_preset_details(preset)
            self._update_status_strip()

    def populate_preset_details(self, preset_data):
        """Populate the preset details view with the given data."""
        details_container = self.query_one("#preset-details-container")
        details_container.remove_children()

        for key, value in preset_data.items():
            if key in {"name", "preset_name", "model", "timestamp", "input_file"}:
                continue
            details_container.mount(Label(f"{key.replace('_', ' ').title()}:"))
            if isinstance(value, list):
                details_container.mount(
                    Input(value=",".join(map(str, value)), id=f"input-{key}")
                )
            else:
                details_container.mount(Input(value=str(value), id=f"input-{key}"))

        run_button_container = Horizontal(
            Button("Run Dream", id="run-dream", variant="primary"),
            id="run-button-container",
        )
        details_container.mount(run_button_container)

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Called when a history row is selected."""
        event.stop()

        file_stem = event.row_key.value
        history_entry = self.history_data.get(file_stem)
        if not history_entry:
            return

        self.current_preset_name = history_entry["preset_name"]
        self.populate_preset_details(history_entry)

        model_radioset = self.query_one(RadioSet)
        model_radioset.pressed_index = 0 if history_entry["model"] == "vgg16" else 1

        output_image_path = Path("outputs") / f"{file_stem}.jpg"
        if output_image_path.exists():
            self._show_output_preview(output_image_path)
        self._update_status_strip(output_image_path)

    def run_dream(self):
        """Run the DeepDream script with the current settings."""
        log = self.query_one("#log")
        if not self.selected_file:
            log.write_line("[bold red]No input file selected.[/bold red]")
            return

        log.clear()
        log.write_line(f"Starting dream for {self.selected_file}...")

        preset = self._get_current_preset()
        if not preset:
            log.write_line("[bold red]No preset selected.[/bold red]")
            return

        settings = self._get_settings_from_inputs(preset)
        model = self.query_one(RadioSet).pressed_button.label.plain
        settings["model"] = model
        settings["input_file"] = str(self.selected_file)
        settings["preset_name"] = self.current_preset_name

        cmd, out_path, json_path = self._build_command(settings)
        settings["timestamp"] = out_path.stem.split("_")[-1]

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(settings, json_file, indent=4)

        log.write_line(f"Command: {' '.join(cmd)}")
        self._execute_dream_process(cmd, out_path)

    def _execute_dream_process(self, cmd, out_path):
        """Execute the DeepDream subprocess and stream output to the log."""
        log = self.query_one("#log")
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            ) as process:
                for line in process.stdout:
                    log.write_line(line.strip())

            log.write_line(
                f"\n[bold green]Dream finished. Saved to {out_path}[/bold green]"
            )
            self.update_history_table()
            self._show_output_preview(out_path)
            self._update_status_strip(out_path)
        except FileNotFoundError:
            log.write_line(
                "[bold red]Error: 'python' command not found."
                " Is it in your PATH?[/bold red]"
            )
        except Exception as error:  # pylint: disable=broad-except
            log.write_line(f"[bold red]An error occurred: {error}[/bold red]")

    def _get_current_preset(self):
        """Return the currently selected preset."""
        return next(
            (
                preset
                for preset in PRESETS
                if preset["name"] == self.current_preset_name
            ),
            None,
        )

    def _get_settings_from_inputs(self, preset):
        """Read input widgets back into a settings dict matching the preset."""
        settings = {}
        for key, original_value in preset.items():
            if key == "name":
                continue
            input_widget = self.query_one(f"#input-{key}", Input)
            value = input_widget.value
            if isinstance(original_value, int):
                value = int(value)
            elif isinstance(original_value, float):
                value = float(value)
            elif isinstance(original_value, list):
                value = value.split(",")
            settings[key] = value
        return settings

    def _build_command(self, settings):
        """Build the command to run DeepDream."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_name = (
            f"{Path(settings['input_file']).stem}_tui_{settings['model']}_"
            f"{settings['preset_name']}_{timestamp}"
        )
        out_path = Path("outputs") / f"{out_name}.jpg"
        json_path = Path("outputs") / f"{out_name}.json"

        cmd = [
            "python",
            "dream.py",
            "--input",
            str(settings["input_file"]),
            "--output",
            str(out_path),
            "--model",
            settings["model"],
        ]
        for key, value in settings.items():
            arg_name = ARG_MAPPING.get(key)
            if not arg_name:
                continue
            if key == "layers":
                cmd.extend([f"--{arg_name}", *value])
            else:
                cmd.extend([f"--{arg_name}", str(value)])
        return cmd, out_path, json_path

    def _set_preview_placeholder(self, widget_id: str, message: str):
        """Show a friendly placeholder message in a preview widget."""
        widget = self.query_one(widget_id, ImagePreview)
        widget.update(message)

    def _auto_select_first_input(self):
        """Auto-select the first image in the input directory to surface a preview."""
        input_dir = Path("input")
        if not input_dir.exists():
            return

        image_files = sorted(
            [
                p
                for p in input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
            ]
        )
        if not image_files:
            return

        self.selected_file = image_files[0]
        self._show_input_preview(self.selected_file)
        self.update_history_table()
        self._show_latest_output_for_input()
        self._update_status_strip()

    def _show_input_preview(self, path: Path):
        """Render the selected input image in the input preview."""
        if not path:
            return
        self.query_one("#input-preview", ImagePreview).update_image(str(path))

    def _show_output_preview(self, path: Path):
        """Render the selected output image in the output preview."""
        if not path:
            return
        self.latest_output_path = path
        self.query_one("#output-preview", ImagePreview).update_image(str(path))

    def _show_latest_output_for_input(self):
        """If outputs exist for the selected input, show the newest one."""
        if not self.history_data:
            return
        newest_key = next(iter(self.history_data))
        newest_output = Path("outputs") / f"{newest_key}.jpg"
        if newest_output.exists():
            self._show_output_preview(newest_output)

    def _update_status_strip(self, output_path: Optional[Path] = None):
        """Update the status strip with the current selections."""
        output_display = output_path or self.latest_output_path
        status = (
            f"Input: {Path(self.selected_file).name if self.selected_file else '—'} | "
            f"Preset: {self.current_preset_name or '—'} | "
            f"Model: {self.query_one(RadioSet).pressed_button.label.plain if self.query_one(RadioSet).pressed_button else '—'} | "
            f"Output: {Path(output_display).name if output_display else '—'}"
        )
        self.query_one("#status-strip", Static).update(status)


if __name__ == "__main__":
    Tui().run()

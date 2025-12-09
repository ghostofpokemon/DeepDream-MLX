"""Custom widgets for the DeepDream TUI."""
import shutil
import subprocess
import base64
import os

from textual.widgets import DirectoryTree, Static
from term_image.image import from_file, ITerm2Image
from PIL import Image as PILImage


# pylint: disable=too-many-ancestors
class FilteredDirectoryTree(DirectoryTree):
    """A DirectoryTree that filters for image files."""

    def filter_paths(self, paths):
        """Filter paths to only include image files and directories."""
        return [
            path
            for path in paths
            if path.is_dir()
            or path.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]


class SvgStatic(Static):
    """A widget to display SVG content."""

    _SVG_CACHE = {}

    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        if path not in self._SVG_CACHE:
            with open(self.path, "r", encoding="utf-8") as f:
                self._SVG_CACHE[path] = f.read()
        self.svg_content = self._SVG_CACHE[path]

    def render(self):
        return self.svg_content


class ImagePreview(Static):
    """A widget to display an image preview."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = None
        self.last_path = None

    def update_image(self, path: str):
        """Update the image preview with the image at the given path."""
        self.last_path = path
        
        # 1. Try imgcat (User Preference)
        # 2. Try term-image (iTerm2, Kitty, Sixel, etc.)
        # 3. Fallback to TrueColor ASCII (The "Art")

        try:
            # Try to use term-image with a simpler approach
            self.image = from_file(path)
            # Use the string representation which should work
            display_text = str(self.image)
            self.update(display_text)
        except Exception:
            # Fallback to Shared Renderer
            from deepdream.ascii_art import render_image_to_string
            
            # Textual provides self.size, passes it to the renderer
            w = max(self.size.width - 2, 20) if self.size.width else None
            h = max(self.size.height - 2, 10) if self.size.height else None
            
            art = render_image_to_string(path, width=w, height=h)
            self.update(art)

    def on_resize(self, event):
        """Called when the widget is resized."""
        if self.last_path:
            self.update_image(self.last_path)

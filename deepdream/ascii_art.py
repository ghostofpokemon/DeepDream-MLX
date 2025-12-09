"""
Console Art & Image Rendering Utilities.
Supports TrueColor (Half-Block), iTerm2, and Imgcat protocols.
"""
import os
import shutil
import base64
import subprocess
from PIL import Image as PILImage

def render_image_to_string(path: str, width: int = None, height: int = None) -> str:
    """
    Render an image to a string suitable for console display.
    Prioritizes: imgcat > iTerm2 > TrueColor Half-Block.
    """
    # 1. Try imgcat (User Preference)
    imgcat_out = _render_imgcat(path, width, height)
    if imgcat_out:
        return imgcat_out

    # 2. Try iTerm2 Native
    if _is_iterm2():
        iterm_out = _render_iterm2_native(path, width, height)
        if iterm_out:
            return iterm_out

    # 3. Fallback to TrueColor ASCII (The "Art")
    return _render_halfblock(path, width)

def _is_iterm2() -> bool:
    """Check if we're running in iTerm2 or compatible terminal."""
    term_program = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")
    return term_program == "iTerm.app" or "iterm" in term.lower()

def _render_iterm2_native(path: str, width: int = None, height: int = None) -> str | None:
    """Render image using native iTerm2 inline image protocol."""
    try:
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            encoded = base64.b64encode(image_data).decode('ascii')

        # Default dimensions if not provided
        w_str = f"{width}px" if width else "auto"
        h_str = f"{height}px" if height else "auto"

        # If width/height not specified, let terminal handle it or default to something sane?
        # Textual passed specific widget sizes. Here we might want "auto" or "100%".
        # For inline=1, width/height are optional.
        
        dims = ""
        if width: dims += f";width={width}px"
        if height: dims += f";height={height}px"

        # iTerm2 inline image protocol
        iterm_sequence = f"\033]1337;File=inline=1{dims}:{encoded}\a"
        return iterm_sequence
    except Exception:
        return None

def _render_imgcat(path: str, width: int = None, height: int = None) -> str | None:
    """Render via imgcat (iTerm-compatible) if available."""
    if not shutil.which("imgcat"):
        return None
    try:
        cmd = ["imgcat"]
        if width:
            cmd.extend(["--width", str(width)])
        if height:
            cmd.extend(["--height", str(height)])
        
        cmd.append(path)

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return None

def _render_halfblock(path: str, target_width: int = None) -> str:
    """Render high-quality TrueColor ASCII art (Half-Block)."""
    try:
        img = PILImage.open(path).convert("RGB")
        
        # Default width if not provided
        if not target_width:
            # Detect terminal size?
            try:
                term_w, _ = os.get_terminal_size()
                target_width = min(term_w, 80) # Cap at 80 for sanity
            except:
                target_width = 60
        
        # Aspect ratio preservation
        w, h = img.size
        aspect = h / w
        
        target_height_chars = int(target_width * aspect * 0.5)
        target_height_px = target_height_chars * 2
        
        img = img.resize((target_width, target_height_px), PILImage.Resampling.BILINEAR)
        data = img.load()
        
        lines = []
        for y in range(0, target_height_px, 2):
            row = []
            for x in range(target_width):
                # Top pixel
                r1, g1, b1 = data[x, y]
                # Bottom pixel
                r2, g2, b2 = data[x, y+1] if y+1 < target_height_px else (0,0,0)
                
                # TrueColor ANSI: 
                # FG (Bottom pixel) = \033[38;2;R;G;Bm
                # BG (Top pixel) = \033[48;2;R;G;Bm
                # Character = ▄ (U+2584)
                
                pixel = f"\033[38;2;{r2};{g2};{b2}m\033[48;2;{r1};{g1};{b1}m▄\033[0m"
                row.append(pixel)
            lines.append("".join(row))
        
        return "\n".join(lines)

    except Exception as e:
        return f"[Image Rendering Failed: {e}]"

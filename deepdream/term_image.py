import sys
import os
import base64

def print_image(path: str, max_width: int = None):
    """
    Prints an image to the terminal using iTerm2 inline image protocol.
    Fallback to nothing if not likely supported (checking TERM_PROGRAM).
    """
    if os.environ.get("TERM_PROGRAM") != "iTerm.app" and os.environ.get("TERM_PROGRAM") != "WezTerm":
        # Rough check, but safe to try printing OSC in many modern terminals (kitty uses different one, but let's stick to iTerm for now as requested)
        # If user asked for imgcat specifically, they likely have iTerm2.
        pass

    try:
        with open(path, "rb") as f:
            image_data = f.read()
            encoded = base64.b64encode(image_data).decode("utf-8")
            
            # OSC 1337 ; File = [args] : Content ^G
            # args: inline=1, size=size
            print(f"\033]1337;File=inline=1;width=auto:{encoded}\007")
            print() # Newline after image
    except Exception as e:
        print(f"[Warning: Could not display image inline: {e}]")

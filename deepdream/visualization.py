
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

# --- Aesthetics ---

class FuturistPalette:
    # From colorsToUse.txt
    CHARCOAL = (46, 56, 46)      # #2e382e
    CYAN = (80, 201, 206)        # #50c9ce
    BLUE = (114, 161, 229)       # #72a1e5
    PURPLE = (152, 131, 229)     # #9883e5
    PINK = (252, 211, 222)       # #fcd3de
    
    @staticmethod
    def rgb(r, g, b):
        return f"\033[38;2;{r};{g};{b}m"
    
    @classmethod
    def gradient(cls, depth, max_depth=10):
        # Interpolate between Cyan and Purple
        t = min(depth / max_depth, 1.0)
        r = int(cls.CYAN[0] * (1-t) + cls.PURPLE[0] * t)
        g = int(cls.CYAN[1] * (1-t) + cls.PURPLE[1] * t)
        b = int(cls.CYAN[2] * (1-t) + cls.PURPLE[2] * t)
        return cls.rgb(r, g, b)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

def print_header(title):
    c_cyan = FuturistPalette.rgb(*FuturistPalette.CYAN)
    c_blue = FuturistPalette.rgb(*FuturistPalette.BLUE)
    c_pink = FuturistPalette.rgb(*FuturistPalette.PINK)
    
    # Neoclassical Border
    print(f"\n{c_cyan}╔{'═'*70}╗{RESET}")
    print(f"{c_cyan}║{' '*70}║{RESET}")
    print(f"{c_cyan}║ {c_pink}{BOLD}{title.center(68)}{RESET}{c_cyan} ║{RESET}")
    print(f"{c_cyan}║ {c_blue}{ITALIC}{'FUTURIST • BEAUX-ARTS • ARCHITECTURE'.center(68)}{RESET}{c_cyan} ║{RESET}")
    print(f"{c_cyan}║{' '*70}║{RESET}")
    print(f"{c_cyan}╠{'═'*70}╣{RESET}")

def print_footer(params):
    c_cyan = FuturistPalette.rgb(*FuturistPalette.CYAN)
    c_purp = FuturistPalette.rgb(*FuturistPalette.PURPLE)
    
    msg = f"TOTAL PARAMETERS: {params:,.0f} ({params/1e6:.2f} M)"
    
    print(f"{c_cyan}╠{'═'*70}╣{RESET}")
    print(f"{c_cyan}║ {c_purp}{BOLD}{msg}{RESET}{' '*(69 - len(msg))} {c_cyan}║{RESET}")
    print(f"{c_cyan}╚{'═'*70}╝{RESET}\n")

# --- Nodes and Neurons Logic ---

def traverse_model(model, model_name):
    """
    Traverses the model and prints a 'nodes and neurons' style visualization.
    """
    seen = set()

    def _walk(mod, prefix="", depth=0, is_last=True):
        if id(mod) in seen: return
        seen.add(id(mod))
        
        # Aesthetics
        indent = "  " * depth

        # Handle Lists/Tuples specifically
        if isinstance(mod, (list, tuple)):
            # Print the container node
            marker = "└── " if is_last else "├── "
            if depth == 0: marker = ""
            c_node = FuturistPalette.gradient(depth)
            print(f"{indent}{c_node}{marker}○ {BOLD}{prefix}{RESET} {DIM}[{type(mod).__name__}]{RESET}")
            
            # Recurse items
            for i, item in enumerate(mod):
                _walk(item, f"[{i}]", depth + 1, is_last=(i == len(mod)-1))
            return
        
        # Node Symbols (Neurons)
        # ○ = Module/Container
        # ● = Leaf Layer (Conv, Linear)
        # ◉ = Weighted Layer
        
        has_weights = False
        try:
            for v in vars(mod).values():
                if isinstance(v, mx.array):
                    has_weights = True
                    break
        except: pass

        symbol = "○"
        if has_weights:
            symbol = "◉"
        
        # Connectors
        marker = "└── " if is_last else "├── "
        if depth == 0: marker = ""
        
        c_node = FuturistPalette.gradient(depth)
        c_dim = DIM
        c_val = FuturistPalette.rgb(*FuturistPalette.PINK)
        c_weight = FuturistPalette.rgb(*FuturistPalette.PURPLE)
        
        name_str = f"{prefix}" if prefix else "ROOT"
        type_str = f"[{mod.__class__.__name__}]"
        
        print(f"{indent}{c_node}{marker}{symbol} {BOLD}{name_str}{RESET} {c_dim}{type_str}{RESET}")

        # 1. Local Parameters
        local_params = []
        try:
            # vars() only works if object has __dict__
            if hasattr(mod, "__dict__"):
                for k, v in vars(mod).items():
                    if isinstance(v, mx.array):
                        local_params.append((k, v))
        except: pass
            
        for pk, pv in sorted(local_params, key=lambda x: x[0]):
            sz = pv.size
            shp = str(pv.shape)
            print(f"{indent}  {c_dim}• {pk}:{RESET} {c_val}{shp}{RESET} {c_weight}({sz:,}){RESET}")

        # 2. Children
        children_items = []
        if hasattr(mod, "children"):
             children_items = list(mod.children().items())
        elif hasattr(mod, "__dict__"):
             for k, v in vars(mod).items():
                 if isinstance(v, nn.Module):
                     children_items.append((k, v))
                 elif isinstance(v, (list, tuple)):
                     for i, item in enumerate(v):
                         if isinstance(item, nn.Module):
                             children_items.append((f"{k}[{i}]", item))
        
        children_items.sort(key=lambda x: str(x[0]))

        for i, (ck, cv) in enumerate(children_items):
            _walk(cv, ck, depth + 1, is_last=(i == len(children_items)-1))

    _walk(model, model_name, 0)

# text_label_node.py
# AddTextLabel — true 50% transparent box via overlay, text fully opaque, centered

import os, platform
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

def _rgba(hexstr: str, alpha: float = 1.0) -> Tuple[int, int, int, int]:
    s = hexstr.strip().lstrip("#")
    if len(s) == 3:
        r, g, b = [int(c * 2, 16) for c in s]
    elif len(s) == 6:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    else:
        r, g, b = 255, 255, 255
    return (r, g, b, max(0, min(255, int(alpha * 255))))

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def _pil_to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(im).astype(np.float32) / 255.0)

def _font_candidates(fam: str) -> List[str]:
    c = []
    if fam == "Arial":
        c += ["Arial.ttf","arial.ttf","/Library/Fonts/Arial.ttf",
              "C:\\Windows\\Fonts\\arial.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"]
    elif fam == "Monospace":
        c += ["DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/Library/Fonts/Menlo.ttc",
              "C:\\Windows\\Fonts\\consola.ttf"]
    if platform.system().lower() == "linux":
        home = os.path.expanduser("~")
        c += [os.path.join(home, ".fonts", "DejaVuSansMono.ttf"),
              os.path.join(home, ".local", "share", "fonts", "DejaVuSansMono.ttf")]
    return c

def _load_font(fam: str, size: int):
    from PIL import ImageFont
    for p in _font_candidates(fam):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font):
    try:
        return draw.textbbox((0, 0), text, font=font)  # (l,t,r,b)
    except Exception:
        w = int(draw.textlength(text, font=font)); h = int(font.size)
        return (0, 0, w, h)

def _place_xy(w, h, bw, bh, placement, edge):
    if placement == "top_left":          x, y = edge, edge
    elif placement == "top_right":       x, y = w - bw - edge, edge
    elif placement == "bottom_left":     x, y = edge, h - bh - edge
    elif placement == "bottom_right":    x, y = w - bw - edge, h - bh - edge
    else:                                x, y = (w - bw) // 2, (h - bh) // 2
    return int(x), int(y)

class AddTextLabel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "YOUR TEXT", "multiline": False}),
            },
            "optional": {
                "font_family": (["Arial", "Monospace"], {"default": "Arial"}),
                "font_size": ("INT", {"default": 30, "min": 6, "max": 256, "step": 1}),
                "placement": (["top_left","top_right","bottom_left","bottom_right","center"], {"default": "top_left"}),
                "edge_offset": ("INT", {"default": 30, "min": 0, "max": 4096, "step": 1}),
                "color_scheme": (["white_on_black","black_on_white"], {"default": "black_on_white"}),
                "padding": ("INT", {"default": 15, "min": 0, "max": 256, "step": 1}),
                "corner_radius": ("INT", {"default": 15, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_label"
    CATEGORY = "image/annotation"

    def add_label(
        self,
        image: torch.Tensor,
        text: str,
        font_family="Arial",
        font_size=30,
        placement="top_left",
        edge_offset=30,
        color_scheme="black_on_white",
        padding=15,
        corner_radius=15,
    ):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        font = _load_font(font_family, int(font_size))
        if color_scheme == "white_on_black":
            text_col, bg_col = "#FFFFFF", "#000000"
        else:
            text_col, bg_col = "#000000", "#FFFFFF"

        out = []
        for i in range(image.shape[0]):
            base = _tensor_to_pil(image[i]).convert("RGBA")
            draw = ImageDraw.Draw(base, "RGBA")

            # Measure text
            l, t, r, b = _text_bbox(draw, text, font)
            tw, th = (r - l), (b - t)

            pad = int(padding)
            box_w, box_h = int(tw + pad * 2), int(th + pad * 2)
            x, y = _place_xy(base.width, base.height, box_w, box_h, placement, int(edge_offset))

            # --- draw 50% transparent box on separate overlay, then composite ---
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay, "RGBA")
            rx = max(0, min(int(corner_radius), min(box_w, box_h) // 2))
            odraw.rounded_rectangle([x, y, x + box_w, y + box_h],
                                    radius=rx,
                                    fill=_rgba(bg_col, 0.5))  # 50% alpha
            base = Image.alpha_composite(base, overlay)

            # Draw text fully opaque on the composited image
            draw = ImageDraw.Draw(base, "RGBA")
            cx, cy = x + box_w / 2.0, y + box_h / 2.0
            tx, ty = int(cx - tw / 2.0 - l), int(cy - th / 2.0 - t)
            draw.text((tx, ty), text, font=font, fill=_rgba(text_col, 1.0))

            out.append(_pil_to_tensor(base.convert("RGB")))

        return (torch.stack(out, dim=0),)

NODE_CLASS_MAPPINGS = {"AddTextLabel": AddTextLabel}
NODE_DISPLAY_NAME_MAPPINGS = {"AddTextLabel": "Add Text Label"}

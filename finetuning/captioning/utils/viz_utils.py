import os
from PIL import Image, ImageDraw, ImageFont

FONT = "resources/fonts/DMSans-Regular.ttf"
COLORS = [
    "black",
    "darkred",
    "darkblue",
    "darkgreen",
    "darksalmon",
    "darkgoldenrod",
    "deeppink",
]


def generate_text(texts, size, fontsize=16):
    """Generate a piece of canvas and draw text on it"""
    canvas = Image.new("RGB", size, "white")

    if isinstance(texts, str):
        texts = [texts]

    # Get a drawing context
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, fontsize)
    margin = size[0] // 12
    position = (margin, size[1] // 12)

    for c, text in enumerate(texts):
        lines = []
        words = text.split()
        i = 0
        while i < len(words):
            cur_text = ""
            while i < len(words) and draw.textlength(cur_text + words[i], font=font) < (
                size[0] - 2 * margin + 5
            ):
                cur_text += " " + words[i]
                i += 1
            if cur_text:
                lines.append(cur_text.strip())

        text = "\n".join(lines)
        draw.text(position, text, font=font, fill=COLORS[c])
        text_bl = draw.textbbox(position, text, font)
        position = (text_bl[0], text_bl[3] + margin // 3)

    return canvas


def viz_frame(frame_idx, prompts, src_path, fontsize=16, c="", outpath=""):
    frame_idx = "0" * (4 - len(str(frame_idx))) + str(frame_idx)
    if not c:
        filename = os.path.join(src_path, f"img_{frame_idx}.jpg")
    else:
        filename = os.path.join(src_path, c, f"img_{frame_idx}.jpg")

    image = Image.open(filename)

    panel = Image.new("RGB", (2 * image.size[0], image.size[1]), "white")
    panel.paste(image)
    panel.paste(generate_text(prompts, image.size, fontsize), (image.size[0], 0))

    if outpath:
        panel.save(outpath)

    return panel

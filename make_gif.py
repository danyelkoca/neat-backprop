import imageio
from PIL import Image, ImageDraw, ImageFont
import os

# List of result PNGs in order
frames = [
    "results/xor_gen_0.png",
    "results/xor_gen_1.png",
    "results/xor_gen_2.png",
    "results/xor_gen_3.png",
    "results/xor_gen_4.png",
    "results/xor_gen_5.png",
    "results/xor_gen_10.png",
    "results/xor_gen_16.png",
    "results/xor_gen_17.png",
    "results/xor_gen_29.png",
    "results/xor_gen_30.png",
    "results/xor_gen_32.png",
]


def add_generation_text(image_path, generation):
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font_size = int(img.height * 0.05)  # reduced to 7% of image height
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)

    text = f"Generation {generation}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - text_width) // 2
    y = int(img.height * 0.05)

    padding_x = int(font_size * 0)
    padding_y = int(font_size * 0)
    rect_x0 = x - padding_x
    rect_y0 = y - padding_y
    rect_x1 = x + text_width + padding_x
    rect_y1 = y + text_height + padding_y
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white", outline=None)
    draw.text((x, y), text, font=font, fill="black")
    return img


images = []
for path in frames:
    gen = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    img = add_generation_text(path, gen)
    images.append(img)

images[0].save(
    "xor_evolution.gif",
    save_all=True,
    append_images=images[1:],
    duration=500,
    loop=0,
    disposal=2,
)
print("GIF saved as xor_evolution.gif")

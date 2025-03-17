import numpy as np
from PIL import Image
from config import (
    IMG_CHANNELS,
    IMG_HEIGHT
    )

def load_and_pad_image(path, target_size = IMG_HEIGHT, to_RGB = IMG_CHANNELS == 3):
    """
    - Opens the image in grayscale
    - If it's larger than target_size in any dimension, resizes down
    - Pads with black so final shape = target_size x target_size
    - Returns a float32 array in [0,1], shape (target_size, target_size, IMG_CHANNELS)
    """
    img = Image.open(path).convert("L")
    w , h = img.size

    ratio = min(target_size / w, target_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (target_size, target_size))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img, (left, top))

    img_np = np.array(new_img, dtype=np.float32) / 255.0

    if to_RGB:
        img_np = np.stack((img_np,)*3, axis=-1)
    else:
        img_np = np.expand_dims(img_np, axis=-1)

    return img_np
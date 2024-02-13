import numpy as np
from PIL import Image


def render_env(T, ds, render_path, save_path, name: str):
    frames = []
    for frame_number in np.arange(0, T):
        frame = Image.open(rf"{render_path}\plot_{frame_number}" + ".png")
        frames.append(frame)

    frames[0].save(
        f"{save_path}/{name}.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=20,
        loop=0,
    )

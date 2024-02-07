from PIL import Image
import numpy as np
import hydra
from omegaconf.dictconfig import DictConfig


def render_env(T, ds, render_path, save_path, name: str):
    frames = []
    length = int(T) // ds
    for frame_number in np.arange(length):
        frame = Image.open(rf"{render_path}\plot_{frame_number}.png")
        frames.append(frame)

    frames[0].save(
        f"{save_path}/{name}.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=50,
        loop=0,
    )

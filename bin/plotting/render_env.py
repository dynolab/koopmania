from PIL import Image
import numpy as np
import hydra

frames = []

for frame_number in np.arange(0, 1601):
    frame = Image.open(
        rf"C:\Users\mWX1298408\Documents\GitHub\koopmania\bin\render_dmd\plot_{frame_number}"
        + ".png"
    )
    frames.append(frame)

frames[0].save(
    f"./render_dmd/ns_view_large.gif",
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=50,
    loop=0,
)

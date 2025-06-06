import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

frames_dir = 'frames_gif_ex'

# Collect sorted file paths
file_paths = sorted(
    [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.startswith('data_') and f.endswith('.h5')],
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)

# take the first hald of these files
# file_paths = file_paths[:len(file_paths) // 2]

# Prepare a list to store each frame image
images = []

for file_path in file_paths:
    # Read data
    with h5py.File(file_path, 'r') as f:
        phi = f["phi"][:]
        X1 = f["X1"][:]
        X2 = f["X2"][:]
        a = f["a"][:]
        b = f["b"][:]
        # Coordinates (assumes X, Y defined globally or regenerate here)
        Ny, Nx = phi.shape
        x = np.linspace(0, 1, Nx)  # adjust if domain differs
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)

    # Mask solid region
    a_masked = np.ma.masked_where(phi <= 0, a)
    b_masked = np.ma.masked_where(phi <= 0, b)
    u_mag_masked = np.sqrt(a**2 + b**2)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    cf = ax.contourf(X, Y, u_mag_masked, levels=50, cmap="Spectral_r")
    ax.contourf(X, Y, phi <= 0, levels=[0.5, 1], colors='white', zorder=2)
    X1_masked = np.where(phi <= 0, X1, np.nan)
    X2_masked = np.where(phi <= 0, X2, np.nan)
    ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=1.5, zorder=3)
    ax.contour(X, Y, X1_masked, levels=15, colors='black', linewidths=0.5, linestyles='solid', zorder=4)
    ax.contour(X, Y, X2_masked, levels=15, colors='black', linewidths=0.5, linestyles='dashed', zorder=4)
    ax.set_title("Deformation of Soft Disc in Lid-Driven Cavity Flow", fontsize=10)
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Convert plot to image array
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    images.append(image)
    plt.close(fig)

# Save as GIF
# current_dir = os.getcwd()
# gif_path = os.path.join(current_dir, 'lid_driven_256x256.gif')
# imageio.mimsave(gif_path, images, fps=50 )

# Save as MP4 video
# video_path = os.path.join(os.getcwd(), 'lid_driven_256x256.mp4')
# with imageio.get_writer(video_path, fps=50, codec='libx264') as writer:
#     for image in images:
#         writer.append_data(image)

import imageio_ffmpeg

video_path = os.path.join(os.getcwd(), 'lid_driven_256x256.mp4')
writer = imageio.get_writer(
    video_path,
    fps=50,
    codec='libx264',
    quality=10,  # 10 is the best for imageio
    macro_block_size=None,  # allow arbitrary frame sizes
    ffmpeg_params=[
        "-preset", "slow",       # better compression at the cost of speed
        "-crf", "18",            # lower = better quality, 18 is visually lossless
        "-pix_fmt", "yuv420p"    # ensures compatibility with most players
    ]
)

for image in images:
    writer.append_data(image)
writer.close()



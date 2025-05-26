import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

# Directory containing frame HDF5 files
frames_dir = 'frames_gif_ex'

# Collect sorted file paths
file_paths = sorted(
    [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.startswith('data_') and f.endswith('.h5')],
    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
)

# Prepare a list to store each frame image
images = []

for file_path in file_paths:
    # Read data
    with h5py.File(file_path, 'r') as f:
        phi = f["phi"][:]
        X1 = f["X1"][:]
        X2 = f["X2"][:]
        a = f["a"][:]
        # Coordinates (assumes X, Y defined globally or regenerate here)
        Ny, Nx = phi.shape
        x = np.linspace(0, 1, Nx)  # adjust if domain differs
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)

    # Mask solid region
    a_masked = np.ma.masked_where(phi <= 0, a)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cf = ax.contourf(X, Y, a_masked, levels=70, cmap="turbo")
    ax.contourf(X, Y, phi <= 0, levels=[0.5, 1], colors='white', zorder=2)
    X1_masked = np.where(phi <= 0, X1, np.nan)
    X2_masked = np.where(phi <= 0, X2, np.nan)
    ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=3.5, zorder=3)
    ax.contour(X, Y, X1_masked, levels=15, colors='black', linewidths=0.75, linestyles='solid', zorder=4)
    ax.contour(X, Y, X2_masked, levels=15, colors='black', linewidths=0.75, linestyles='dashed', zorder=4)
    ax.set_title("Deformed Mesh via Reference Map Contours")
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Convert plot to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close(fig)

# Save as GIF
current_dir = os.getcwd()
gif_path = os.path.join(current_dir, 'lid_driven.gif')
imageio.mimsave(gif_path, images, fps=20)


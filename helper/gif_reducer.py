# Re-import necessary libraries after the reset
from PIL import Image, ImageSequence

# Define the input and output paths
input_path = "lid_driven_256x256_new_2.gif"
output_path = "lid_driven_half_duration.gif"

# Open the original GIF
original_gif = Image.open(input_path)

# Extract all frames
frames = [frame.copy() for frame in ImageSequence.Iterator(original_gif)]
original_duration = original_gif.info.get('duration', 50)  # Default to 100ms per frame

# Keep only the first half of the frames
half_index = len(frames) // 2
half_frames = frames[:half_index]

# Reduce the duration of each frame (speed up the GIF)
new_duration = max(original_duration // 2, 1)

# Save the new GIF
half_frames[0].save(
    output_path,
    save_all=True,
    append_images=half_frames[1:],
    loop=0,
    duration=new_duration,
    disposal=2
)

output_path

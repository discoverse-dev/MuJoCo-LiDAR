import argparse
import os

import cv2
import numpy as np


# Mock SubTerrain to match terrain_utils interface
class SubTerrain:
    def __init__(self, width, length, vertical_scale, horizontal_scale):
        self.width = width
        self.length = length
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0, base_platform=0.0):
    """
    Generate stairs
    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        base_platform (float): size of the flat platform at the base (edge) of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    base_platform = int(base_platform / terrain.horizontal_scale)

    height = 0
    start_x = base_platform
    stop_x = terrain.width - base_platform
    start_y = base_platform
    stop_y = terrain.length - base_platform

    # If step_height is negative, we are going down.
    # The logic adds step_height.
    # So if step_height is negative, it goes 0 -> -H -> -2H ...

    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height

        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width

    return terrain


def main():
    parser = argparse.ArgumentParser(description="Generate random pyramid stairs terrain grid")
    parser.add_argument(
        "--resolution", type=int, default=500, help="Image resolution (width/height)"
    )
    parser.add_argument(
        "--size", type=float, default=12.0, help="Physical size of terrain in meters"
    )
    parser.add_argument("--grid_size", type=int, default=2, help="NxN grid")
    parser.add_argument("--step_height", type=float, default=0.15, help="Height of each step (m)")
    parser.add_argument("--num_steps", type=int, default=7, help="Number of steps per pyramid")
    parser.add_argument("--platform_size", type=float, default=1.0, help="Size of top platform (m)")
    parser.add_argument("--pad_meters", type=float, default=1.5, help="Extend on all sides (m)")
    parser.add_argument(
        "--base_platform", type=float, default=1.0, help="Size of base platform (m)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/generated_terrain", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parameters
    width = args.resolution
    length = args.resolution
    horizontal_scale = args.size / args.resolution
    vertical_scale = 0.001  # 1mm
    pad_meters = args.pad_meters

    # Initialize full heightfield
    # Use int16 for accumulation
    full_hfield = np.zeros((width, length), dtype=np.int16)

    # Grid parameters
    n = args.grid_size
    sub_w = width // n
    sub_l = length // n

    # Step width calculation
    # Each sub-terrain has size sub_w * horizontal_scale (meters)
    # We want num_steps.
    # The pyramid goes from edge to center.
    # Formula: 2 * base_platform + 2 * num_steps * step_width + platform_size = sub_w_m

    # User request: "Shorten distance between pyramids by half".
    # Originally distance is 2 * base_platform. We use base_platform / 2 to make gap = 1 * base_platform.
    effective_base_platform = args.base_platform / 1.5

    sub_w_m = sub_w * horizontal_scale
    available_run = (sub_w_m - args.platform_size - 2 * effective_base_platform) / 2

    if available_run <= 0:
        print("Warning: No space for steps! Increase size or decrease platforms.")
        step_width_m = 0.1  # fallback
    else:
        step_width_m = available_run / args.num_steps

    step_width_pixels = int(step_width_m / horizontal_scale)

    print(f"Generating {n}x{n} grid.")
    print(
        f"Sub-terrain size: {sub_w}x{sub_l} pixels ({sub_w * horizontal_scale:.2f}m x {sub_l * horizontal_scale:.2f}m)"
    )
    print(f"Step width: {step_width_pixels} pixels ({step_width_m:.2f}m)")
    print(f"Step height: {args.step_height}m")
    print(f"Base platform (original): {args.base_platform}m, Effective: {effective_base_platform}m")
    print(f"Top platform: {args.platform_size}m")

    # np.random.seed(42)

    for i in range(n):
        for j in range(n):
            # Randomly choose up or down
            is_up = np.random.random() > 0.5
            h = args.step_height if is_up else -args.step_height

            # Create sub-terrain
            sub = SubTerrain(sub_w, sub_l, vertical_scale, horizontal_scale)
            pyramid_stairs_terrain(
                sub,
                step_width_m,
                h,
                platform_size=args.platform_size,
                base_platform=effective_base_platform,
            )

            # Place in full hfield
            # Handle edge cases if resolution not divisible by n
            r_start = i * sub_w
            r_end = r_start + sub_w
            c_start = j * sub_l
            c_end = c_start + sub_l

            # Ensure we don't overflow if grid doesn't divide perfectly
            if r_end > width:
                r_end = width
            if c_end > length:
                c_end = length

            # Crop sub if needed
            sub_h_crop = sub.height_field_raw[: r_end - r_start, : c_end - c_start]

            full_hfield[r_start:r_end, c_start:c_end] = sub_h_crop

    pad_pixels = int(pad_meters / horizontal_scale)

    print(f"Padding: {pad_meters}m ({pad_pixels} pixels) on each side.")

    new_width = width + 2 * pad_pixels
    new_length = length + 2 * pad_pixels
    padded_hfield = np.zeros((new_width, new_length), dtype=np.int16)

    # Place original hfield in center
    padded_hfield[pad_pixels : pad_pixels + width, pad_pixels : pad_pixels + length] = full_hfield

    # Update full_hfield reference
    full_hfield = padded_hfield
    print(full_hfield.max(), full_hfield.min())

    # Update physical dimensions for MJCF
    # Use actual pixel dimensions to preserve horizontal_scale exactly
    total_size_x = new_width * horizontal_scale
    total_size_y = new_length * horizontal_scale

    # Normalize and Save
    min_h_raw = np.min(full_hfield)
    max_h_raw = np.max(full_hfield)

    print(f"Raw height range: {min_h_raw} to {max_h_raw} (units of {vertical_scale}m)")
    print(f"Physical range: {min_h_raw * vertical_scale:.3f}m to {max_h_raw * vertical_scale:.3f}m")

    # Shift to positive
    shifted_hfield = full_hfield - min_h_raw
    range_raw = max_h_raw - min_h_raw

    if range_raw == 0:
        range_raw = 1  # Avoid div by zero

    # Scale to 0-255 for 8-bit PNG
    output_img = (shifted_hfield / range_raw * 255).astype(np.uint8)

    img_path = os.path.join(args.output_dir, "terrain.png")
    cv2.imwrite(img_path, output_img)
    print(f"Saved image to {img_path}")

    # Generate MJCF
    mjcf_path = os.path.join(args.output_dir, "terrain.xml")

    rx = total_size_x / 2
    ry = total_size_y / 2
    ez = range_raw * vertical_scale
    bz = 0.1  # Base z

    # Position offset to align "0" level
    # The "0" level in raw data corresponds to value -min_h_raw in shifted data.
    # Normalized value = (-min_h_raw) / range_raw
    # Physical height of "0" level in geom = Normalized * ez = -min_h_raw * vertical_scale
    # We want this to be at World Z=0.
    # So we must shift geom by - (Physical height of "0" level)
    # pos_z = - (-min_h_raw * vertical_scale) = min_h_raw * vertical_scale

    # Correction: geom pos is the CENTER of the bounding box.
    # The hfield geom extends from [Z_data_0 - bz, Z_data_0 + ez].
    # The center is at Z_data_0 + (ez - bz) / 2.
    # We want Z_data_0 to be at min_h_raw * vertical_scale.
    # So pos_z = (min_h_raw * vertical_scale) + (ez - bz) / 2

    pos_z = min_h_raw * vertical_scale + (ez - bz) / 2

    mjcf_content = f"""<mujoco model="generated_terrain">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <hfield name="terrain" file="terrain.png" size="{rx} {ry} {ez} {bz}" />
  </asset>

  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {pos_z}"/>
  </worldbody>
</mujoco>
"""
    with open(mjcf_path, "w") as f:
        f.write(mjcf_content)
    print(f"Saved MJCF to {mjcf_path}")


if __name__ == "__main__":
    main()

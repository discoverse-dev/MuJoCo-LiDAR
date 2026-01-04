import argparse
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.cm as cm

def decompose_mask_to_rects(mask):
    """
    Decompose a binary mask into a list of non-overlapping rectangles.
    Returns list of (r, c, h, w).
    """
    rects = []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                # Start new rect
                r0, c0 = r, c
                
                # Find max width for this row
                c1 = c0 + 1
                while c1 < w and mask[r0, c1] and not visited[r0, c1]:
                    c1 += 1
                width = c1 - c0
                
                # Find max height for this width
                r1 = r0 + 1
                while r1 < h:
                    # Check if the next row segment is all True and unvisited
                    row_slice = mask[r1, c0:c1]
                    visited_slice = visited[r1, c0:c1]
                    if np.all(row_slice) and not np.any(visited_slice):
                        r1 += 1
                    else:
                        break
                height = r1 - r0
                
                # Mark visited
                visited[r0:r1, c0:c1] = True
                rects.append((r0, c0, height, width))
    return rects

def merge_rects(rects):
    """
    Iteratively merge adjacent rectangles.
    rects: list of (r, c, h, w)
    """
    while True:
        merged_any = False
        new_rects = []
        used = [False] * len(rects)
        
        for i in range(len(rects)):
            if used[i]: continue
            
            r1, c1, h1, w1 = rects[i]
            merged_i = False
            
            for j in range(i + 1, len(rects)):
                if used[j]: continue
                
                r2, c2, h2, w2 = rects[j]
                
                # Check vertical adjacency (same width, same column, adjacent rows)
                if c1 == c2 and w1 == w2:
                    if r1 + h1 == r2: # r2 is directly below r1
                        rects[j] = (r1, c1, h1 + h2, w1)
                        used[i] = True
                        merged_i = True
                        merged_any = True
                        break
                    elif r2 + h2 == r1: # r1 is directly below r2
                        rects[j] = (r2, c2, h1 + h2, w1)
                        used[i] = True
                        merged_i = True
                        merged_any = True
                        break
                        
                # Check horizontal adjacency (same height, same row, adjacent columns)
                if r1 == r2 and h1 == h2:
                    if c1 + w1 == c2: # c2 is directly right of c1
                        rects[j] = (r1, c1, h1, w1 + w2)
                        used[i] = True
                        merged_i = True
                        merged_any = True
                        break
                    elif c2 + w2 == c1: # c1 is directly right of c2
                        rects[j] = (r1, c2, h1, w1 + w2)
                        used[i] = True
                        merged_i = True
                        merged_any = True
                        break
            
            if not merged_i:
                new_rects.append(rects[i])
        
        # If we merged something, the 'rects' list contains updated merged rects (at index j)
        # and we skipped adding 'i' to new_rects.
        # However, we need to be careful: 'new_rects' only contains unmerged 'i's.
        # The merged results are sitting in 'rects[j]'.
        # So we need to collect:
        # 1. Items that were NOT used (not merged into anything, and nothing merged into them? No.)
        # Actually, the logic above is: if i is merged into j, i is marked used, j is updated.
        # Later we visit j. j might be merged into k.
        # So we just need to collect all items that are NOT used at the end of the pass?
        # No, because we iterate i from 0 to N.
        # If i merges into j (j>i), i is used. j is updated.
        # When we reach j, it is treated as a fresh rect.
        # So we only need to collect:
        # - rects[i] if i was never merged into anything (used[i] == False).
        # - BUT, if k merged into i (k<i), then i IS used? No, k is used. i is updated.
        # So 'used' means "this rect has been absorbed by another rect to its right/bottom".
        # So we just collect all rects[i] where not used[i].
        
        rects = [rects[k] for k in range(len(rects)) if not used[k]]
        
        if not merged_any:
            break
            
    return rects

def main():
    parser = argparse.ArgumentParser(description='Convert MuJoCo Hfield to Box Geoms')
    parser.add_argument('--xml', type=str, required=True, help='Input MJCF file path')
    parser.add_argument('--hfield', type=str, required=True, help='Name of the hfield asset')
    parser.add_argument('--output', type=str, default='converted_hfield.xml', help='Output MJCF file path')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum height to generate a box (m)')
    parser.add_argument('--merge_threshold', type=float, default=0.05, help='Merge height levels within this threshold (m)')
    parser.add_argument('--colormap', type=str, default='viridis', help='Matplotlib colormap for boxes')
    args = parser.parse_args()
    
    print(f"Loading model from {args.xml}...")
    try:
        m = mujoco.MjModel.from_xml_path(args.xml)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    # Find hfield
    hfield_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, args.hfield)
    if hfield_id == -1:
        print(f"Error: Hfield '{args.hfield}' not found in model.")
        # List available hfields
        n_hfield = m.nhfield
        if n_hfield > 0:
            print("Available hfields:")
            for i in range(n_hfield):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, i)
                print(f"  - {name}")
        else:
            print("No hfields found in model.")
        return

    print(f"Processing hfield '{args.hfield}' (ID: {hfield_id})...")

    # Get hfield parameters
    nrow = m.hfield_nrow[hfield_id]
    ncol = m.hfield_ncol[hfield_id]
    adr = m.hfield_adr[hfield_id]
    
    # Size: (radius_x, radius_y, elevation_z, base_z)
    size = m.hfield_size[hfield_id]
    rx, ry, ez, bz = size
    
    print(f"Dimensions: {nrow}x{ncol}")
    print(f"Size: rx={rx}, ry={ry}, ez={ez}, bz={bz}")

    # Find geom using this hfield to get offset
    geom_pos = np.array([0.0, 0.0, 0.0])
    geom_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    found_geom = False
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD and m.geom_dataid[i] == hfield_id:
            geom_pos = m.geom_pos[i]
            geom_quat = m.geom_quat[i]
            found_geom = True
            print(f"Found geom {i} using hfield '{args.hfield}'.")
            print(f"  Pos: {geom_pos}")
            print(f"  Quat: {geom_quat}")
            break
            
    if not found_geom:
        print("Warning: No geom found using this hfield. No offset applied.")
    
    # Get data and reshape
    # MuJoCo hfield data is normalized [0, 1]
    raw_data = m.hfield_data[adr:adr+nrow*ncol].reshape(nrow, ncol)
    
    # Convert to real height
    heights = raw_data * ez
    print(heights.max(), heights.min())

    # Smart Quantization (Clustering by frequency)
    if args.merge_threshold > 0:
        print(f"Clustering height levels with threshold {args.merge_threshold}m...")
        unique_vals, counts = np.unique(heights, return_counts=True)
        
        # Sort by value
        sorted_indices = np.argsort(unique_vals)
        unique_vals = unique_vals[sorted_indices]
        counts = counts[sorted_indices]
        
        mapping = {}
        
        i = 0
        while i < len(unique_vals):
            # Start a new cluster
            cluster_vals = [unique_vals[i]]
            cluster_counts = [counts[i]]
            
            j = i + 1
            while j < len(unique_vals):
                # Check if close enough to the cluster start
                if unique_vals[j] - unique_vals[i] < args.merge_threshold:
                    cluster_vals.append(unique_vals[j])
                    cluster_counts.append(counts[j])
                    j += 1
                else:
                    break
            
            # Find representative (mode)
            cluster_vals = np.array(cluster_vals)
            cluster_counts = np.array(cluster_counts)
            best_idx = np.argmax(cluster_counts)
            representative = cluster_vals[best_idx]
            
            for v in cluster_vals:
                mapping[v] = representative
                
            i = j
            
        # Apply mapping
        # Use searchsorted to map heights to indices in unique_vals (which are sorted)
        idx = np.searchsorted(unique_vals, heights)
        # Create lookup table
        # Note: unique_vals contains all original values. mapping maps them to new values.
        lookup = np.array([mapping[v] for v in unique_vals])
        heights = lookup[idx]

    # Find unique levels
    unique_levels = np.unique(heights)
    # Filter out near-zero heights (ground)
    unique_levels = unique_levels[unique_levels > args.threshold]
    unique_levels.sort()
    
    print(f"Found {len(unique_levels)} unique height levels (above {args.threshold}m).")
    if len(unique_levels) > 100:
        print(f"Warning: Large number of levels ({len(unique_levels)}). This may generate many geoms.")

    # Grid spacing
    # The grid spans [-rx, rx] and [-ry, ry]
    # dx is the spacing between columns
    dx = 2 * rx / (ncol - 1) if ncol > 1 else 2 * rx
    dy = 2 * ry / (nrow - 1) if nrow > 1 else 2 * ry
    
    # Initialize XML
    root = ET.Element('mujoco', {'model': f'converted_{args.hfield}'})
    
    # Add visual settings
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', {'diffuse': '0.6 0.6 0.6', 'ambient': '0.3 0.3 0.3', 'specular': '0 0 0'})
    ET.SubElement(visual, 'rgba', {'haze': '0.15 0.25 0.35 1'})
    ET.SubElement(visual, 'global', {'azimuth': '120', 'elevation': '-20'})

    # Add default class
    default = ET.SubElement(root, 'default')
    default_class = ET.SubElement(default, 'default', {'class': 'terrain_box'})
    ET.SubElement(default_class, 'geom', {
        'condim': '3',
        'friction': '1 0.005 0.0001',
        'type': 'box'
    })

    worldbody = ET.SubElement(root, 'worldbody')

    # Create terrain body with offset
    terrain_body = ET.SubElement(worldbody, 'body', {
        'name': 'terrain_body',
        'pos': f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}",
        'quat': f"{geom_quat[0]} {geom_quat[1]} {geom_quat[2]} {geom_quat[3]}"
    })
    
    # Add Ground Plane
    ET.SubElement(terrain_body, 'geom', {
        'name': 'ground_plane',
        'type': 'plane',
        'size': f"{rx*1.5} {ry*1.5} 0.1",
        'pos': '0 0 0',
        'rgba': '0.3 0.3 0.3 1'
    })
    
    # Generate Boxes using Layered Approach
    box_count = 0
    prev_level = 0.0
    
    cmap = cm.get_cmap(args.colormap)
    max_h = unique_levels[-1] if len(unique_levels) > 0 else 1.0
    
    for level in unique_levels:
        layer_thickness = level - prev_level
        
        # Skip negligible layers
        if layer_thickness < 1e-6:
            continue
            
        # Mask: all areas that reach at least this level
        mask = heights >= (level - 1e-6)
        
        # Decompose mask into rectangles
        rects = decompose_mask_to_rects(mask)
        # Merge adjacent rectangles
        rects = merge_rects(rects)
        
        # Z parameters for this layer
        # The box sits on top of the previous level
        # Center Z = prev_level + half_thickness
        pos_z = prev_level + layer_thickness / 2.0
        size_z = layer_thickness / 2.0
        
        # Color based on current level (top of this box)
        rgba = cmap(level / max_h)
        rgba_str = f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1"
        
        for (r, c, h, w) in rects:
            # Calculate center X, Y
            # Grid starts at -rx, -ry
            # Index c corresponds to x = -rx + c*dx
            # Rect covers indices [c, c+w-1]
            # Center index = c + (w-1)/2
            
            cx_idx = c + (w - 1) / 2.0
            cy_idx = r + (h - 1) / 2.0
            
            pos_x = -rx + cx_idx * dx
            pos_y = -ry + cy_idx * dy
            
            # Size (half-extents)
            # Full width = w * dx
            # But wait, does a grid point represent a point or a cell?
            # In MuJoCo hfield, vertices are at grid points.
            # If we want to fill the volume, we should treat each grid point as the center of a cell of size dx*dy?
            # Or treat the quad between (r,c) and (r+1,c+1) as the cell?
            # Given "convert hfield to geom", usually we want to approximate the volume.
            # Treating points as cell centers is a standard voxelization approach.
            # So full size X = w * dx.
            
            size_x = (w * dx) / 2.0
            size_y = (h * dy) / 2.0
            
            ET.SubElement(terrain_body, 'geom', {
                'class': 'terrain_box',
                'pos': f"{pos_x:.4f} {pos_y:.4f} {pos_z:.4f}",
                'size': f"{size_x:.4f} {size_y:.4f} {size_z:.4f}",
                'rgba': rgba_str
            })
            box_count += 1
            
        prev_level = level
        
    print(f"Generated {box_count} box geoms.")
    print(f"Saving to {args.output}...")
    
    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
    with open(args.output, 'w') as f:
        f.write(xml_str)
    
    print("Done.")

if __name__ == '__main__':
    main()

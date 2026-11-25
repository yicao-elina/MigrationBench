#!/usr/bin/env python3
"""
Render trajectory frames to publication-quality PNG images using POV-Ray

Usage:
    python render_trajectory.py <input_file> [frame_slice] [options]

Examples:
    python render_trajectory.py sb2te3.xyz            # Render all frames
    python render_trajectory.py sb2te3.xyz :          # Render all frames
    python render_trajectory.py sb2te3.xyz ::2        # Every 2nd frame
    python render_trajectory.py sb2te3.xyz 0          # First frame only
    python render_trajectory.py sb2te3.xyz -1         # Last frame only
    python render_trajectory.py sb2te3.xyz 1:5        # Frames 1-4
    python render_trajectory.py sb2te3.xyz 1:10:2     # Frames 1,3,5,7,9
Color examples:
    python render_traj_color.py  2L_441_octo_Cr1_v2.xyz --colors "Te:0.8,0.7,0.4;Sb:0.9,0.6,0.4;Cr:0.7,0.6,0.9" -r 270x,30y,0z  
    python render_trajectory.py sb2te3.xyz --colors "C:0.3,0.3,0.3;H:1.0,1.0,1.0"
"""

import sys
import os
import numpy as np
from ase.io import read, write
import argparse

def parse_frame_slice(frame_str):
    """Parse frame slice string and return slice object or int"""
    if frame_str is None:
        return slice(None)  # All frames
    
    # Single frame
    if ':' not in frame_str:
        return int(frame_str)
    
    # Slice notation
    parts = frame_str.split(':')
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return slice(start, stop)
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else None
        return slice(start, stop, step)
    else:
        raise ValueError(f"Invalid frame slice: {frame_str}")

def parse_color_string(color_str):
    """
    Parse color string format: "Element1:r,g,b;Element2:r,g,b;..."
    Returns dict: {'Element1': [r, g, b], 'Element2': [r, g, b], ...}
    """
    if not color_str:
        return None
    
    color_dict = {}
    try:
        # Split by semicolon to get individual element definitions
        element_colors = color_str.split(';')
        for element_color in element_colors:
            if not element_color.strip():
                continue
            # Split by colon to separate element from RGB values
            element, rgb_str = element_color.strip().split(':')
            # Parse RGB values
            r, g, b = map(float, rgb_str.split(','))
            # Ensure values are in [0, 1] range
            if not all(0 <= val <= 1 for val in [r, g, b]):
                raise ValueError(f"RGB values must be between 0 and 1 for element {element}")
            color_dict[element.strip()] = [r, g, b]
    except Exception as e:
        raise ValueError(f"Invalid color string format: {color_str}. Error: {str(e)}")
    
    return color_dict

def render_frames(input_file, frame_slice=None, output_prefix=None, **kwargs):
    """Render specified frames from trajectory file"""
    
    # Read trajectory
    print(f"Reading trajectory from {input_file}...")
    trajectory = read(input_file, index=':')  # Read all frames first
    
    # Handle frame selection
    if isinstance(frame_slice, int):
        # Single frame
        frames_to_render = [trajectory[frame_slice]]
        frame_indices = [frame_slice if frame_slice >= 0 else len(trajectory) + frame_slice]
    else:
        # Slice
        frames_to_render = trajectory[frame_slice]
        # Calculate actual frame indices
        start, stop, step = frame_slice.indices(len(trajectory))
        frame_indices = list(range(start, stop, step))
    
    # Ensure frames_to_render is a list
    if not isinstance(frames_to_render, list):
        frames_to_render = [frames_to_render]
    
    print(f"Total frames in trajectory: {len(trajectory)}")
    print(f"Frames to render: {len(frames_to_render)} (indices: {frame_indices})")
    
    # Output prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_file))[0]
    
    # Parse custom colors if provided
    custom_color_dict = kwargs.get('color_dict', None)
    
    # Render each frame
    for i, (frame_idx, atoms) in enumerate(zip(frame_indices, frames_to_render)):
        output_base = f"{output_prefix}_frame_{frame_idx:04d}"
        pov_file = f"{output_base}.pov"
        png_file = f"{output_base}.png"
        
        print(f"\nRendering frame {frame_idx} ({i+1}/{len(frames_to_render)})...")
        
        # Rotation angles - can be customized per frame if needed
        rot = kwargs.get('rotation', '270x,30y,0z')
        
        # High-quality rendering settings
        generic_projection_settings = {
            'rotation': rot,
            'radii': kwargs.get('radii', 0.9),
            'colors': None,
            'show_unit_cell': kwargs.get('show_unit_cell', 0),
        }
        
        # Publication-quality POV-Ray settings
        povray_settings = {
            'display': kwargs.get('display', False),
            'pause': False,
            'transparent': kwargs.get('transparent', False),
            'canvas_width': kwargs.get('width', 2400),
            # 'canvas_height': kwargs.get('height', 2400),
            'camera_dist': kwargs.get('camera_dist', 50.0),
            'image_plane': None,
            'camera_type': 'perspective',
            'point_lights': [
                # [[30, 30, 50], 'White'],
                # [[-30, 30, 50], [0.6, 0.6, 0.6]],
            ],
            'area_light': [
                (2.0, 3.0, 40.0),
                'White',
                0.7,
                0.7,
                3,
                3,
            ],
            'background': 'Transparent' if kwargs.get('transparent', False) else 'White',
            'textures': None,
            'celllinewidth': 0.05,
        }
        
        # Apply custom colors
        if custom_color_dict:
            custom_colors = []
            for atom in atoms:
                if atom.symbol in custom_color_dict:
                    custom_colors.append(custom_color_dict[atom.symbol])
                else:
                    # Default color for elements not specified
                    custom_colors.append([0.5, 0.5, 0.5])
            generic_projection_settings['colors'] = custom_colors
            
            # Print which elements are being colored
            if i == 0:  # Only print once
                print("Custom colors applied:")
                for element, color in custom_color_dict.items():
                    print(f"  {element}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        elif kwargs.get('custom_colors', True):
            # Default Sb2Te3 colors if no custom colors specified
            custom_colors = []
            for atom in atoms:
                if atom.symbol == 'Sb':
                    custom_colors.append([0.7, 0.7, 0.8])  # Silver-gray
                elif atom.symbol == 'Te':
                    custom_colors.append([0.8, 0.6, 0.3])  # Bronze/gold
                else:
                    custom_colors.append([0.5, 0.5, 0.5])  # Default gray
            generic_projection_settings['colors'] = custom_colors
        
        # Write and render
        renderer = write(
            pov_file,
            atoms,
            **generic_projection_settings,
            povray_settings=povray_settings,
        )
        
        renderer.render()
        print(f"Saved: {png_file}")
        
        # Optionally remove .pov and .ini files
        if kwargs.get('clean', True):
            os.remove(pov_file)
            ini_file = pov_file.replace('.pov', '.ini')
            if os.path.exists(ini_file):
                os.remove(ini_file)

def main():
    parser = argparse.ArgumentParser(
        description='Render trajectory frames to publication-quality PNG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('input_file', help='Input trajectory file (xyz, traj, etc.)')
    parser.add_argument('frame_slice', nargs='?', default=None,
                       help='Frame selection (e.g., ":", "::2", "0", "-1", "1:5", "1:10:2")')
    parser.add_argument('-o', '--output-prefix', help='Output file prefix')
    parser.add_argument('-r', '--rotation', default='30x,45y,0z',
                       help='Rotation angles (default: 30x,45y,0z)')
    parser.add_argument('-w', '--width', type=int, default=2400,
                       help='Image width in pixels (default: 2400)')
    parser.add_argument('-H', '--height', type=int, default=2400,
                       help='Image height in pixels (default: 2400)')
    parser.add_argument('--radii', type=float, default=0.9,
                       help='Atomic radii scaling (default: 0.9)')
    parser.add_argument('--camera-dist', type=float, default=50.0,
                       help='Camera distance (default: 50.0)')
    parser.add_argument('--show-cell', action='store_true',
                       help='Show unit cell')
    parser.add_argument('--transparent', action='store_true',
                       help='Transparent background')
    parser.add_argument('--display', action='store_true',
                       help='Display while rendering')
    parser.add_argument('--no-custom-colors', action='store_true',
                       help='Use default ASE colors (disable custom Sb/Te colors)')
    parser.add_argument('--colors', type=str, default=None,
                       help='Custom colors for elements. Format: "Element1:r,g,b;Element2:r,g,b;..." '
                            'RGB values should be between 0 and 1. '
                            'Example: "Sb:0.7,0.7,0.8;Te:0.8,0.6,0.3"')
    parser.add_argument('--keep-pov', action='store_true',
                       help='Keep .pov and .ini files after rendering')
    
    args = parser.parse_args()
    
    # Parse frame slice
    try:
        frame_slice = parse_frame_slice(args.frame_slice)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse custom colors
    color_dict = None
    if args.colors:
        try:
            color_dict = parse_color_string(args.colors)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Color format should be: 'Element1:r,g,b;Element2:r,g,b;...'", file=sys.stderr)
            print("Example: 'Sb:0.7,0.7,0.8;Te:0.8,0.6,0.3'", file=sys.stderr)
            sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Render frames
    render_frames(
        args.input_file,
        frame_slice=frame_slice,
        output_prefix=args.output_prefix,
        rotation=args.rotation,
        width=args.width,
        height=args.height,
        radii=args.radii,
        camera_dist=args.camera_dist,
        show_unit_cell=2 if args.show_cell else 0,
        transparent=args.transparent,
        display=args.display,
        custom_colors=not args.no_custom_colors,
        color_dict=color_dict,
        clean=not args.keep_pov
    )

if __name__ == "__main__":
    main()
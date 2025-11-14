#!/usr/bin/env python3
"""
Utility script to convert PLY files to FBX format.
Takes a PLY file path as argument and saves the FBX file in the same folder.

Uses Blender's command-line interface for reliable FBX export.
Requires Blender to be installed and accessible via 'blender' command.
"""

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path


def find_blender():
    """Find Blender executable."""
    # Try common Blender paths
    common_paths = [
        'blender',
        '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS default
        '/usr/bin/blender',
        '/usr/local/bin/blender',
    ]
    
    for path in common_paths:
        try:
            result = subprocess.run(
                [path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return path
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    return None


def convert_ply_to_fbx(ply_path: Path, blender_path: str = None) -> Path:
    """
    Convert a PLY file to FBX format using Blender.
    
    Args:
        ply_path: Path to the input PLY file
        blender_path: Path to Blender executable (auto-detected if None)
        
    Returns:
        Path to the output FBX file
        
    Raises:
        FileNotFoundError: If the PLY file doesn't exist or Blender is not found
        ValueError: If the file is not a valid PLY file or conversion fails
    """
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    if not ply_path.suffix.lower() == '.ply':
        raise ValueError(f"Input file must be a .ply file, got: {ply_path.suffix}")
    
    # Find Blender if not provided
    if blender_path is None:
        blender_path = find_blender()
        if blender_path is None:
            raise FileNotFoundError(
                "Blender not found. Please install Blender and ensure it's in your PATH, "
                "or specify the path using --blender_path option."
            )
    
    print(f"[Info] Using Blender: {blender_path}")
    print(f"[Info] Loading PLY file: {ply_path}")
    
    # Generate output path (same folder, same name, different extension)
    fbx_path = ply_path.with_suffix('.fbx')
    
    # Create a temporary Python script for Blender
    blender_script = f"""
import bpy
import sys

# Clear existing mesh data
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import PLY file
try:
    bpy.ops.wm.ply_import(filepath=r"{ply_path}")
    print("[Info] Successfully imported PLY file")
except Exception as e:
    print(f"[Error] Failed to import PLY: {{e}}")
    sys.exit(1)

# Export to FBX
try:
    bpy.ops.export_scene.fbx(
        filepath=r"{fbx_path}",
        use_selection=False,
        apply_scale_options='FBX_SCALE_NONE',
        bake_space_transform=False
    )
    print("[Info] Successfully exported to FBX")
except Exception as e:
    print(f"[Error] Failed to export FBX: {{e}}")
    sys.exit(1)
"""
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(blender_script)
    
    try:
        print(f"[Info] Exporting to FBX: {fbx_path}")
        
        # Run Blender in background mode
        result = subprocess.run(
            [
                blender_path,
                '--background',
                '--python', script_path
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise ValueError(f"Blender conversion failed: {error_msg}")
        
        if not fbx_path.exists():
            raise ValueError("FBX file was not created. Check Blender output for errors.")
        
        print(f"[Info] Successfully converted {ply_path.name} to {fbx_path.name}")
        
        return fbx_path
        
    except subprocess.TimeoutExpired:
        raise ValueError("Conversion timed out after 5 minutes")
    except Exception as e:
        raise ValueError(f"Failed to convert PLY to FBX: {str(e)}") from e
    finally:
        # Clean up temporary script
        try:
            Path(script_path).unlink()
        except Exception:
            pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a PLY file to FBX format using Blender. "
                    "The output FBX file will be saved in the same folder as the input PLY file."
    )
    parser.add_argument(
        'ply_path',
        type=str,
        help="Path to the input PLY file"
    )
    parser.add_argument(
        '--blender_path',
        type=str,
        default=None,
        help="Path to Blender executable (auto-detected if not specified)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        ply_path = Path(args.ply_path).resolve()
        fbx_path = convert_ply_to_fbx(ply_path, blender_path=args.blender_path)
        print(f"[Info] Conversion complete: {fbx_path}")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


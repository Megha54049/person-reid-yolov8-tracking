#!/usr/bin/env python3
"""Fix label format by removing the extra track ID column"""

from pathlib import Path

def fix_labels(label_dir):
    """Remove the 6th column (track ID) from all label files"""
    label_dir = Path(label_dir)
    label_files = list(label_dir.glob("*.txt"))
    
    fixed_count = 0
    for label_file in label_files:
        # Read file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Fix format
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:
                # Remove last column (track ID)
                fixed_line = ' '.join(parts[:5]) + '\n'
                fixed_lines.append(fixed_line)
            elif len(parts) == 5:
                fixed_lines.append(line)
            else:
                print(f"Warning: Unexpected format in {label_file}: {line.strip()}")
        
        # Write fixed file
        with open(label_file, 'w') as f:
            f.writelines(fixed_lines)
        
        fixed_count += 1
        if fixed_count % 50 == 0:
            print(f"Fixed {fixed_count}/{len(label_files)} files...")
    
    print(f"✅ Fixed {fixed_count} label files")

# Fix both train and val
print("Fixing training labels...")
fix_labels("/home/meghaagrawal940/annotated_data/2_stairs_annotated/data/labels/train")

print("\nFixing validation labels...")
fix_labels("/home/meghaagrawal940/annotated_data/2_stairs_annotated/data/labels/val")

print("\n✅ All labels fixed!")

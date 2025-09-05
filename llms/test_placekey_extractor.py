#!/usr/bin/env python3
"""
Extract placekeys from JSONL files - one input file = one output file
"""

import json
import argparse
from pathlib import Path
import glob

def extract_placekeys_from_file(input_file, output_file=None):
    """Extract placekeys from one JSONL file and save to one output file"""
    
    # Determine output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.placekeys.txt')
    
    placekeys = []
    
    print(f"Processing: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'placekey' in data:
                        placekeys.append(data['placekey'])
                        
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON on line {line_num}")
                    continue
        
        # Save placekeys to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for placekey in placekeys:
                f.write(placekey + '\n')
        
        print(f"  -> Saved {len(placekeys)} placekeys to: {output_file}")
        return len(placekeys)
        
    except FileNotFoundError:
        print(f"  Error: File not found: {input_file}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract placekeys: one input file -> one output file")
    parser.add_argument("input_files", nargs="+", help="Input JSONL files (supports wildcards like *.jsonl)")
    parser.add_argument("--suffix", default=".placekeys.txt", help="Output file suffix (default: .placekeys.txt)")
    
    args = parser.parse_args()
    
    # Expand wildcards
    all_files = []
    for pattern in args.input_files:
        if '*' in pattern or '?' in pattern:
            all_files.extend(glob.glob(pattern))
        else:
            all_files.append(pattern)
    
    if not all_files:
        print("No files found!")
        return
    
    print(f"Found {len(all_files)} file(s) to process\n")
    
    total_placekeys = 0
    successful_files = 0
    
    for input_file in all_files:
        # Generate output filename
        input_path = Path(input_file)
        output_file = input_path.with_suffix(args.suffix)
        
        count = extract_placekeys_from_file(input_file, output_file)
        if count > 0:
            successful_files += 1
            total_placekeys += count
    
    print(f"\n=== Summary ===")
    print(f"Files processed successfully: {successful_files}/{len(all_files)}")
    print(f"Total placekeys extracted: {total_placekeys}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Add FIPS county code and county name fields to existing train/test JSONL files
"""

import json
import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Add FIPS and county fields to existing JSONL files")
    p.add_argument("--safegraph", required=True, help="Original SafeGraph CSV file")
    p.add_argument("--jsonl", required=True, nargs='+', help="JSONL files to update (can specify multiple)")
    return p.parse_args()

def create_placekey_lookup(safegraph_csv):
    """Create placekey -> (fips, county_name) lookup from SafeGraph data"""
    logger.info(f"Loading SafeGraph data from {safegraph_csv}")
    
    sg_df = pd.read_csv(safegraph_csv)
    
    # Create lookup dictionary
    lookup = {}
    for _, row in sg_df.iterrows():
        placekey = row['placekey']
        fips_code = int(row['fips_full_county']) if pd.notna(row['fips_full_county']) else None
        county_name = row.get('county_name', None)
        
        if placekey not in lookup:
            lookup[placekey] = {
                'fips_county_code': fips_code,
                'county_name': county_name
            }
    
    logger.info(f"Created lookup for {len(lookup)} placekeys")
    return lookup

def update_jsonl_with_fields(jsonl_file, output_file, lookup):
    """Add FIPS and county fields to JSONL records"""
    logger.info(f"Processing {jsonl_file}")
    
    updated_records = []
    missing_placekeys = set()
    
    with open(jsonl_file, 'r', encoding='ascii', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                placekey = record.get('placekey')
                
                if placekey in lookup:
                    # Add the new fields
                    record['fips_county_code'] = lookup[placekey]['fips_county_code']
                    record['county_name'] = lookup[placekey]['county_name']
                else:
                    missing_placekeys.add(placekey)
                    # Add null values for missing placekeys
                    record['fips_county_code'] = None
                    record['county_name'] = None
                
                updated_records.append(record)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue
    
    # Write updated records
    with open(output_file, 'w', encoding='ascii', errors='ignore') as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')
    
    logger.info(f"Updated {len(updated_records)} records")
    if missing_placekeys:
        logger.warning(f"Missing placekeys in SafeGraph data: {len(missing_placekeys)} records")
        logger.warning(f"Sample missing placekeys: {list(missing_placekeys)[:5]}")
    
    return len(updated_records), len(missing_placekeys)

def main():
    args = parse_args()
    
    # Create lookup from SafeGraph data
    lookup = create_placekey_lookup(args.safegraph)
    
    total_files = len(args.jsonl)
    total_records = 0
    total_missing = 0
    
    print(f"Processing {total_files} JSONL files...")
    
    # Process each JSONL file
    for jsonl_file in args.jsonl:
        input_path = Path(jsonl_file)
        
        # Create output filename in same directory as input
        output_filename = f"updated_{input_path.name}"
        output_path = input_path.parent / output_filename
        
        print(f"\nProcessing: {jsonl_file}")
        
        # Update JSONL file
        file_updated, file_missing = update_jsonl_with_fields(
            jsonl_file, 
            output_path, 
            lookup
        )
        
        total_records += file_updated
        total_missing += file_missing
        
        print(f"  -> Output: {output_path}")
        print(f"  -> Records: {file_updated}, Missing: {file_missing}")
    
    print(f"\nAll processing complete!")
    print(f"Total files processed: {total_files}")
    print(f"Total records updated: {total_records}")
    print(f"Total missing placekeys: {total_missing}")
    
    # Show sample from first file
    first_input = Path(args.jsonl[0])
    first_output = first_input.parent / f"updated_{first_input.name}"
    if first_output.exists():
        with open(first_output, 'r', encoding='ascii', errors='ignore') as f:
            first_record = json.loads(f.readline())
            print(f"\nSample updated record from {first_output.name}:")
            print(f"   Placekey: {first_record.get('placekey')}")
            print(f"   FIPS Code: {first_record.get('fips_county_code')}")
            print(f"   County: {first_record.get('county_name')}")

if __name__ == "__main__":
    main()
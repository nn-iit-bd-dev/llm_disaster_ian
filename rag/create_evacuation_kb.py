#!/usr/bin/env python3
"""
Evacuation Knowledge Base for RAG System
Creates searchable knowledge base from evacuation orders
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Create evacuation knowledge base")
    p.add_argument("--evacuation", required=True, help="Evacuation CSV file")
    p.add_argument("--output", default="evacuation_kb.json", help="Output knowledge base JSON")
    p.add_argument("--hurricane-name", default="Ian", help="Hurricane name to filter")
    return p.parse_args()

def create_evacuation_knowledge_base(evacuation_csv, hurricane_name="Ian"):
    """Create structured knowledge base from evacuation data"""
    logger.info(f"Loading evacuation data from {evacuation_csv}")
    
    evac_df = pd.read_csv(evacuation_csv)
    
    # Filter for specific hurricane
    if hurricane_name:
        evac_df = evac_df[evac_df['Event Name'].str.contains(hurricane_name, case=False, na=False)]
    
    # Parse dates
    evac_df['announcement_date'] = pd.to_datetime(evac_df['Announcement Date'], errors='coerce')
    evac_df['effective_date'] = pd.to_datetime(evac_df['Effective Date'], errors='coerce')
    
    knowledge_entries = []
    
    for idx, row in evac_df.iterrows():
        county_text = str(row.get('County', '')).strip()
        county_fips = str(row.get('County FIPS', '')).strip()
        
        # Skip if no FIPS code
        if not county_fips or county_fips.lower() in ['nan', 'null', '']:
            continue
        
        # Handle multiple FIPS codes (comma-separated)
        fips_codes = []
        try:
            if ',' in county_fips:
                # Multiple FIPS codes
                fips_list = county_fips.split(',')
                for fips in fips_list:
                    fips = fips.strip()
                    if fips:
                        fips_codes.append(int(fips))
            else:
                # Single FIPS code
                fips_codes.append(int(county_fips))
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing FIPS codes in row {idx}: {e}")
            continue
        
        # Create entry for each FIPS code
        for fips_code in fips_codes:
            # Determine scope
            if ('entire state' in county_text.lower() or 
                'state' in county_text.lower() or 
                ',' in county_text or  # Multiple counties listed
                'counties' in county_text.lower()):
                scope = 'statewide'
                priority = 2
            else:
                scope = 'county_specific'
                priority = 1
            
            # Create searchable text
            searchable_text = f"""
            Hurricane {hurricane_name} evacuation order for FIPS {fips_code} ({county_text}).
            Order type: {row.get('Order Type', 'Unknown')}.
            Announced on {row.get('Announcement Date', 'Unknown')} and effective {row.get('Effective Date', 'Unknown')}.
            Evacuation area: {row.get('Evacuation Area', 'Not specified')}.
            Scope: {scope} order.
            """.strip()
            
            knowledge_entry = {
                'id': f"evac_{idx}_{fips_code}",
                'fips_code': fips_code,
                'county_text': county_text,
                'event_name': row.get('Event Name', ''),
                'order_type': row.get('Order Type', ''),
                'order_type_code': int(row.get('Order Type Code', 0)) if pd.notna(row.get('Order Type Code')) else 0,
                'announcement_date': row['announcement_date'].strftime('%Y-%m-%d') if pd.notna(row['announcement_date']) else None,
                'effective_date': row['effective_date'].strftime('%Y-%m-%d') if pd.notna(row['effective_date']) else None,
                'evacuation_area': row.get('Evacuation Area', ''),
                'scope': scope,
                'priority': priority,
                'searchable_text': searchable_text,
                'source_row': idx
            }
            
            knowledge_entries.append(knowledge_entry)
    
    knowledge_base = {
        'metadata': {
            'hurricane_name': hurricane_name,
            'total_entries': len(knowledge_entries),
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': evacuation_csv
        },
        'entries': knowledge_entries
    }
    
    logger.info(f"Created knowledge base with {len(knowledge_entries)} entries")
    return knowledge_base

class EvacuationKB:
    """RAG Knowledge Base for Evacuation Orders"""
    
    def __init__(self, kb_json_path):
        with open(kb_json_path, 'r', encoding='ascii', errors='ignore') as f:
            self.kb_data = json.load(f)
        self.entries = self.kb_data['entries']
        logger.info(f"Loaded knowledge base with {len(self.entries)} entries")
    
    def retrieve_by_fips(self, fips_code, max_results=5):
        """Retrieve evacuation orders by FIPS code"""
        if not fips_code:
            return []
        
        # Find exact FIPS matches
        matches = [entry for entry in self.entries if entry['fips_code'] == fips_code]
        
        # Sort by priority (county-specific first) then by date
        matches.sort(key=lambda x: (x['priority'], x['announcement_date'] or '9999-12-31'))
        
        return matches[:max_results]
    
    def retrieve_by_date_range(self, target_date, days_before=30, days_after=7):
        """Retrieve orders within date range of target date"""
        target_dt = pd.to_datetime(target_date)
        start_date = (target_dt - pd.Timedelta(days=days_before)).strftime('%Y-%m-%d')
        end_date = (target_dt + pd.Timedelta(days=days_after)).strftime('%Y-%m-%d')
        
        relevant = []
        for entry in self.entries:
            announcement = entry.get('announcement_date')
            if announcement and start_date <= announcement <= end_date:
                relevant.append(entry)
        
        return relevant
    
    def get_context_for_location(self, fips_code, target_date, max_context=3):
        """Get relevant evacuation context for a specific location and date"""
        if not fips_code:
            return []
        
        # Get orders for this FIPS code
        fips_orders = self.retrieve_by_fips(fips_code)
        
        # Filter by date relevance
        target_dt = pd.to_datetime(target_date)
        relevant_orders = []
        
        for order in fips_orders:
            announcement_date = order.get('announcement_date')
            if announcement_date:
                announcement_dt = pd.to_datetime(announcement_date)
                # Include orders announced before target date
                if announcement_dt <= target_dt:
                    # Calculate days between announcement and target
                    days_diff = (target_dt - announcement_dt).days
                    order['days_since_announcement'] = days_diff
                    
                    # Calculate effective date difference if available
                    effective_date = order.get('effective_date')
                    if effective_date:
                        effective_dt = pd.to_datetime(effective_date)
                        order['days_since_effective'] = (target_dt - effective_dt).days
                        order['order_active'] = effective_dt <= target_dt
                    else:
                        order['days_since_effective'] = days_diff
                        order['order_active'] = True
                    
                    relevant_orders.append(order)
        
        # Sort by priority and recency
        relevant_orders.sort(key=lambda x: (x['priority'], -x['days_since_announcement']))
        
        return relevant_orders[:max_context]
    
    def format_context_text(self, context_orders):
        """Format evacuation orders into readable context text"""
        if not context_orders:
            return "No evacuation orders found for this location."
        
        context_parts = []
        for order in context_orders:
            text = f"Evacuation: {order['order_type']}"
            
            if order.get('days_since_announcement', 0) >= 0:
                text += f" (announced {order['days_since_announcement']} days ago"
                
                if 'days_since_effective' in order and order['days_since_effective'] != order['days_since_announcement']:
                    text += f", effective {order['days_since_effective']} days ago"
                text += ")"
            
            if order.get('evacuation_area'):
                text += f". Area: {order['evacuation_area']}"
            
            context_parts.append(text)
        
        return " | ".join(context_parts)

def main():
    args = parse_args()
    
    # Create knowledge base
    kb = create_evacuation_knowledge_base(args.evacuation, args.hurricane_name)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='ascii', errors='ignore') as f:
        json.dump(kb, f, indent=2, ensure_ascii=True)
    
    logger.info(f"Knowledge base saved to {output_path}")
    
    # Test the knowledge base
    print(f"\nTesting knowledge base:")
    kb_test = EvacuationKB(output_path)
    
    # Test with a sample FIPS code (Lee County = 12071)
    test_fips = 12071
    test_date = "2022-10-02"
    context = kb_test.get_context_for_location(test_fips, test_date)
    context_text = kb_test.format_context_text(context)
    
    print(f"Sample context for FIPS {test_fips} on {test_date}:")
    print(f"  {context_text}")

if __name__ == "__main__":
    main()
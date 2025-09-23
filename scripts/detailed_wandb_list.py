#!/usr/bin/env python3
"""
Detailed script to list wandb runs with full information for inspection.
"""

import sys
from datetime import datetime

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)

def list_detailed_runs():
    """List runs with detailed information."""
    
    entity = "anton-safarevich-arc-virtual-cell-challenge"
    project = "state"
    
    print(f"Fetching detailed information for runs in {entity}/{project}")
    print("This may take a moment...")
    
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        runs_list = list(runs)
        
        print(f"\nFound {len(runs_list)} runs")
        print("=" * 120)
        print(f"{'#':<3} | {'ID':<10} | {'Name':<35} | {'Created':<19} | {'Status':<10} | {'Tags':<20} | {'Duration':<10}")
        print("-" * 120)
        
        for i, run in enumerate(runs_list, 1):
            try:
                # Parse creation time
                created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                
                # Get tags
                tags_str = ', '.join(run.tags) if run.tags else 'None'
                if len(tags_str) > 20:
                    tags_str = tags_str[:17] + '...'
                
                # Get status
                status = getattr(run, 'state', 'unknown')
                
                # Get duration
                duration = 'N/A'
                if hasattr(run, 'summary') and run.summary:
                    if 'runtime' in run.summary:
                        duration = f"{run.summary['runtime']:.1f}s"
                    elif '_wandb' in run.summary and 'runtime' in run.summary['_wandb']:
                        duration = f"{run.summary['_wandb']['runtime']:.1f}s"
                
                # Truncate name if too long
                name = run.name[:32] + '...' if len(run.name) > 35 else run.name
                
                print(f"{i:<3} | {run.id:<10} | {name:<35} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status:<10} | {tags_str:<20} | {duration:<10}")
                
            except Exception as e:
                print(f"{i:<3} | {run.id:<10} | ERROR: {e}")
        
        print("\n" + "=" * 120)
        print(f"Total: {len(runs_list)} runs")
        
        # Show some statistics
        print(f"\nStatistics:")
        print(f"- Runs from today: {sum(1 for run in runs_list if run.created_at.startswith('2025-09-16'))}")
        print(f"- Runs from yesterday: {sum(1 for run in runs_list if run.created_at.startswith('2025-09-15'))}")
        print(f"- Runs from this week: {sum(1 for run in runs_list if run.created_at.startswith('2025-09-1'))}")
        
        return runs_list
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    runs = list_detailed_runs()
    if not runs:
        sys.exit(1)

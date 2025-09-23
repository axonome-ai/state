#!/usr/bin/env python3
"""
Simple script to list wandb runs for inspection.
"""

import sys
import os
from datetime import datetime

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)


def list_runs(project: str, entity: str, limit: int = 100):
    """List runs in a project for inspection."""
    
    # Check if we have an API key
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("Warning: WANDB_API_KEY not found in environment variables.")
        print("You may need to run 'wandb login' first.")
    
    try:
        api = wandb.Api()
        print(f"Fetching runs from {entity}/{project}...")
        
        # Try to get runs with error handling
        runs = api.runs(f"{entity}/{project}", per_page=limit)
        runs = list(runs)
        
        if not runs:
            print(f"No runs found in project {entity}/{project}")
            return
        
        print(f"\nFound {len(runs)} runs in project {entity}/{project}")
        print("=" * 100)
        print(f"{'#':<3} | {'ID':<12} | {'Name':<40} | {'Created':<19} | {'Status':<10} | {'Tags'}")
        print("-" * 100)
        
        for i, run in enumerate(runs, 1):
            try:
                created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                tags_str = ', '.join(run.tags) if run.tags else 'None'
                name = run.name[:38] + '..' if len(run.name) > 40 else run.name
                status = getattr(run, 'state', 'unknown')
                print(f"{i:<3} | {run.id:<12} | {name:<40} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status:<10} | {tags_str}")
            except Exception as e:
                print(f"{i:<3} | {run.id:<12} | ERROR: {e}")
            
    except Exception as e:
        print(f"Error accessing project {entity}/{project}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged in: wandb login")
        print("2. Check your internet connection")
        print("3. Verify the project and entity names are correct")
        print("4. Check if you have access to this project")
        sys.exit(1)


if __name__ == "__main__":
    # Default values from your config
    entity = "anton-safarevich-arc-virtual-cell-challenge"
    project = "state"
    
    print(f"Listing runs from {entity}/{project}")
    print("Make sure you're logged in with: wandb login")
    print()
    
    list_runs(project, entity, limit=200)

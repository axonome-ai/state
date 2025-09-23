#!/usr/bin/env python3
"""
Very simple script to try to list wandb runs with minimal dependencies.
"""

import sys
import time

try:
    import wandb
    print("✓ wandb imported successfully")
except ImportError as e:
    print(f"✗ Error importing wandb: {e}")
    sys.exit(1)

def try_list_runs():
    """Try to list runs with retry logic."""
    
    entity = "anton-safarevich-arc-virtual-cell-challenge"
    project = "state"
    
    print(f"Attempting to list runs from {entity}/{project}")
    print("This may take a moment...")
    
    for attempt in range(3):
        try:
            print(f"\nAttempt {attempt + 1}/3...")
            
            # Initialize API
            api = wandb.Api()
            print("✓ API initialized")
            
            # Try to get runs
            print("Fetching runs...")
            runs = api.runs(f"{entity}/{project}")
            
            # Convert to list (this is where the actual API call happens)
            runs_list = list(runs)
            print(f"✓ Successfully fetched {len(runs_list)} runs")
            
            if not runs_list:
                print("No runs found in this project.")
                return
            
            # Display first few runs
            print(f"\nFirst 10 runs:")
            print("-" * 80)
            for i, run in enumerate(runs_list[:10], 1):
                print(f"{i:2d}. {run.id} - {run.name}")
            
            if len(runs_list) > 10:
                print(f"... and {len(runs_list) - 10} more runs")
            
            return runs_list
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < 2:  # Don't sleep on last attempt
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("\nAll attempts failed. Possible issues:")
                print("1. Network connectivity problems")
                print("2. Wandb server issues")
                print("3. Authentication problems")
                print("4. Project/entity name incorrect")
                print("5. No access to this project")
                return None

if __name__ == "__main__":
    runs = try_list_runs()
    if runs:
        print(f"\n✓ Successfully listed {len(runs)} runs")
    else:
        print("\n✗ Failed to list runs")
        sys.exit(1)

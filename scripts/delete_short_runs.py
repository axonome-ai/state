#!/usr/bin/env python3
"""
Script to delete wandb runs with less than 5 epochs or no epoch data.
"""

import sys
from datetime import datetime

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)

def get_short_runs(entity: str, project: str, min_epochs: int = 5):
    """Get runs with less than min_epochs or no epoch data."""
    
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        runs_list = list(runs)
        
        print(f"Analyzing {len(runs_list)} runs for epoch count...")
        
        short_runs = []  # All runs with < min_epochs (including 0 epochs and no data)
        
        for i, run in enumerate(runs_list, 1):
            try:
                history = run.history()
                epoch_columns = [col for col in history.columns if 'epoch' in col.lower()]
                
                if epoch_columns:
                    max_epoch = 0
                    for col in epoch_columns:
                        if not history[col].isna().all():
                            max_epoch = max(max_epoch, history[col].max())
                    
                    if max_epoch < min_epochs:
                        short_runs.append((run, max_epoch, "has_epochs"))
                else:
                    # No epoch columns - treat as 0 epochs
                    if len(history) == 0:
                        short_runs.append((run, 0, "no_data"))
                    else:
                        short_runs.append((run, 0, f"no_epochs_{len(history)}_steps"))
                
                if i % 20 == 0:
                    print(f"Processed {i}/{len(runs_list)} runs...")
                    
            except Exception as e:
                # Error accessing run - treat as 0 epochs
                short_runs.append((run, 0, f"error: {str(e)[:50]}"))
        
        return short_runs
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def display_runs(runs, title):
    """Display runs in a nice format."""
    if not runs:
        print(f"No {title.lower()} found.")
        return
        
    print(f"\n{title} ({len(runs)} runs):")
    print("-" * 110)
    print(f"{'#':<3} | {'ID':<10} | {'Name':<35} | {'Epochs':<8} | {'Type':<15} | {'Created':<19} | {'Status'}")
    print("-" * 110)
    
    for i, (run, epochs, run_type) in enumerate(runs, 1):
        created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        status = getattr(run, 'state', 'unknown')
        name = run.name[:32] + '...' if len(run.name) > 35 else run.name
        
        run_type_display = run_type[:14] + '...' if len(run_type) > 15 else run_type
        print(f"{i:<3} | {run.id:<10} | {name:<35} | {epochs:<8} | {run_type_display:<15} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status}")

def delete_runs(runs, dry_run=False):
    """Delete the specified runs."""
    if not runs:
        print("No runs to delete.")
        return
        
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(runs)} runs")
        return
        
    print(f"\nAbout to delete {len(runs)} runs")
    print("This action cannot be undone!")
    
    # Double confirmation
    response1 = input("Are you sure you want to delete these runs? Type 'yes' to continue: ")
    if response1.lower() != 'yes':
        print("Deletion cancelled.")
        return
        
    response2 = input("Type 'DELETE' to confirm deletion: ")
    if response2 != 'DELETE':
        print("Deletion cancelled.")
        return
    
    # Delete runs
    deleted_count = 0
    failed_count = 0
    
    print(f"\nDeleting {len(runs)} runs...")
    for i, (run, _, _) in enumerate(runs, 1):
        try:
            run.delete()
            deleted_count += 1
            print(f"✓ [{i:2d}/{len(runs)}] Deleted {run.id} ({run.name})")
        except Exception as e:
            failed_count += 1
            print(f"✗ [{i:2d}/{len(runs)}] Failed to delete {run.id}: {e}")
    
    print(f"\nDeletion complete: {deleted_count} successful, {failed_count} failed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete wandb runs with less than 5 epochs (including runs with no epoch data)")
    parser.add_argument("--entity", default="anton-safarevich-arc-virtual-cell-challenge", help="Wandb entity")
    parser.add_argument("--project", default="state", help="Wandb project")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum epochs threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--zero-epochs-only", action="store_true", help="Only delete runs with exactly 0 epochs")
    parser.add_argument("--has-epochs-only", action="store_true", help="Only delete runs that have epoch data but < min_epochs")
    
    args = parser.parse_args()
    
    print(f"Finding runs with < {args.min_epochs} epochs in {args.entity}/{args.project}")
    print("(This includes runs with no epoch data, which are treated as 0 epochs)")
    
    # Get short runs
    short_runs = get_short_runs(args.entity, args.project, args.min_epochs)
    
    if not short_runs:
        print("No short runs found.")
        return
    
    # Filter runs based on options
    if args.zero_epochs_only:
        short_runs = [(run, epochs, run_type) for run, epochs, run_type in short_runs if epochs == 0]
    elif args.has_epochs_only:
        short_runs = [(run, epochs, run_type) for run, epochs, run_type in short_runs if run_type == "has_epochs"]
    
    if not short_runs:
        print("No runs found matching the specified criteria.")
        return
    
    # Display runs
    display_runs(short_runs, f"RUNS WITH < {args.min_epochs} EPOCHS")
    
    # Show statistics
    zero_epoch_count = sum(1 for _, epochs, _ in short_runs if epochs == 0)
    has_epochs_count = sum(1 for _, _, run_type in short_runs if run_type == "has_epochs")
    no_data_count = sum(1 for _, _, run_type in short_runs if run_type == "no_data")
    
    print(f"\nSTATISTICS:")
    print(f"- Total runs to delete: {len(short_runs)}")
    print(f"- Runs with 0 epochs (no data): {zero_epoch_count}")
    print(f"- Runs with < {args.min_epochs} epochs (has data): {has_epochs_count}")
    print(f"- Runs with no data at all: {no_data_count}")
    
    # Delete runs
    delete_runs(short_runs, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

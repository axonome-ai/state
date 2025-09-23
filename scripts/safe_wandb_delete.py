#!/usr/bin/env python3
"""
Safe script to delete wandb runs with multiple safety checks and options.
"""

import sys
import argparse
from datetime import datetime, timedelta

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)

def get_runs(entity: str, project: str):
    """Get all runs from the project."""
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        return list(runs)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return None

def filter_runs(runs, status=None, days_old=None, tags=None, name_pattern=None):
    """Filter runs based on criteria."""
    filtered = []
    
    for run in runs:
        # Filter by status
        if status and getattr(run, 'state', 'unknown') not in status:
            continue
            
        # Filter by age
        if days_old:
            created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
            if created_at > datetime.now() - timedelta(days=days_old):
                continue
                
        # Filter by tags
        if tags:
            run_tags = set(run.tags) if run.tags else set()
            if not any(tag in run_tags for tag in tags):
                continue
                
        # Filter by name pattern
        if name_pattern:
            if name_pattern not in run.name:
                continue
                
        filtered.append(run)
    
    return filtered

def display_runs(runs, title="Runs to be deleted"):
    """Display runs in a nice format."""
    if not runs:
        print(f"No runs found matching criteria.")
        return
        
    print(f"\n{title} ({len(runs)} runs):")
    print("=" * 100)
    print(f"{'#':<3} | {'ID':<10} | {'Name':<35} | {'Created':<19} | {'Status':<10} | {'Duration':<10}")
    print("-" * 100)
    
    for i, run in enumerate(runs, 1):
        created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
        status = getattr(run, 'state', 'unknown')
        
        # Get duration
        duration = 'N/A'
        if hasattr(run, 'summary') and run.summary:
            if 'runtime' in run.summary:
                duration = f"{run.summary['runtime']:.1f}s"
            elif '_wandb' in run.summary and 'runtime' in run.summary['_wandb']:
                duration = f"{run.summary['_wandb']['runtime']:.1f}s"
        
        name = run.name[:32] + '...' if len(run.name) > 35 else run.name
        print(f"{i:<3} | {run.id:<10} | {name:<35} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status:<10} | {duration:<10}")

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
    for i, run in enumerate(runs, 1):
        try:
            run.delete()
            deleted_count += 1
            print(f"✓ [{i:2d}/{len(runs)}] Deleted {run.id} ({run.name})")
        except Exception as e:
            failed_count += 1
            print(f"✗ [{i:2d}/{len(runs)}] Failed to delete {run.id}: {e}")
    
    print(f"\nDeletion complete: {deleted_count} successful, {failed_count} failed")

def main():
    parser = argparse.ArgumentParser(
        description="Safely delete wandb runs with multiple safety checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all failed runs (dry run)
  python scripts/safe_wandb_delete.py --status failed --dry-run

  # Delete all failed runs from today
  python scripts/safe_wandb_delete.py --status failed --days-old 0

  # Delete all crashed runs older than 3 days
  python scripts/safe_wandb_delete.py --status crashed --days-old 3

  # Delete runs with specific tags
  python scripts/safe_wandb_delete.py --tags debug,test

  # Delete runs matching name pattern
  python scripts/safe_wandb_delete.py --name-pattern "2025-09-16"

  # Delete specific run IDs
  python scripts/safe_wandb_delete.py --run-ids run1,run2,run3
        """
    )
    
    # Project settings
    parser.add_argument("--entity", default="anton-safarevich-arc-virtual-cell-challenge", help="Wandb entity")
    parser.add_argument("--project", default="state", help="Wandb project")
    
    # Filtering options
    parser.add_argument("--status", nargs="+", help="Filter by status (failed, crashed, finished, running)")
    parser.add_argument("--days-old", type=int, help="Only delete runs older than N days")
    parser.add_argument("--tags", nargs="+", help="Filter by tags (any match)")
    parser.add_argument("--name-pattern", help="Filter by name pattern (substring match)")
    parser.add_argument("--run-ids", help="Comma-separated list of specific run IDs to delete")
    
    # Safety options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--limit", type=int, help="Limit number of runs to process")
    
    args = parser.parse_args()
    
    # Get all runs
    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = get_runs(args.entity, args.project)
    if not runs:
        sys.exit(1)
    
    print(f"Found {len(runs)} total runs")
    
    # Filter runs
    if args.run_ids:
        # Handle specific run IDs
        run_ids = [rid.strip() for rid in args.run_ids.split(',')]
        filtered_runs = [run for run in runs if run.id in run_ids]
        if len(filtered_runs) != len(run_ids):
            found_ids = {run.id for run in filtered_runs}
            missing_ids = set(run_ids) - found_ids
            print(f"Warning: Could not find runs with IDs: {missing_ids}")
    else:
        # Apply filters
        filtered_runs = filter_runs(
            runs, 
            status=args.status,
            days_old=args.days_old,
            tags=args.tags,
            name_pattern=args.name_pattern
        )
    
    # Apply limit if specified
    if args.limit and len(filtered_runs) > args.limit:
        print(f"Limiting to first {args.limit} runs")
        filtered_runs = filtered_runs[:args.limit]
    
    # Display and delete
    display_runs(filtered_runs)
    delete_runs(filtered_runs, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

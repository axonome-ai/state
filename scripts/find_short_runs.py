#!/usr/bin/env python3
"""
Script to find wandb runs with less than 5 epochs.
"""

import sys
from datetime import datetime

try:
    import wandb
except ImportError:
    print("Error: wandb is not installed. Install it with: pip install wandb")
    sys.exit(1)

def analyze_runs(entity: str, project: str, min_epochs: int = 5):
    """Analyze runs to find those with less than min_epochs."""
    
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        runs_list = list(runs)
        
        print(f"Analyzing {len(runs_list)} runs for epoch count...")
        print("This may take a moment...")
        
        short_runs = []
        runs_with_epochs = []
        runs_without_epochs = []
        
        for i, run in enumerate(runs_list, 1):
            try:
                # Get the run's history to find epoch information
                history = run.history()
                
                # Look for epoch-related columns
                epoch_columns = [col for col in history.columns if 'epoch' in col.lower()]
                
                if epoch_columns:
                    # Get the maximum epoch value
                    max_epoch = 0
                    for col in epoch_columns:
                        if not history[col].isna().all():
                            max_epoch = max(max_epoch, history[col].max())
                    
                    runs_with_epochs.append((run, max_epoch))
                    
                    if max_epoch < min_epochs:
                        short_runs.append((run, max_epoch))
                else:
                    # No epoch columns found, check if there are any metrics at all
                    if len(history) == 0:
                        runs_without_epochs.append((run, "no_data"))
                    else:
                        runs_without_epochs.append((run, f"no_epochs_{len(history)}_steps"))
                
                if i % 10 == 0:
                    print(f"Processed {i}/{len(runs_list)} runs...")
                    
            except Exception as e:
                print(f"Error processing run {run.id}: {e}")
                runs_without_epochs.append((run, f"error: {str(e)[:50]}"))
        
        # Display results
        print(f"\n" + "="*80)
        print(f"ANALYSIS RESULTS")
        print(f"="*80)
        print(f"Total runs analyzed: {len(runs_list)}")
        print(f"Runs with epoch data: {len(runs_with_epochs)}")
        print(f"Runs without epoch data: {len(runs_without_epochs)}")
        print(f"Runs with < {min_epochs} epochs: {len(short_runs)}")
        
        if short_runs:
            print(f"\nRUNS WITH < {min_epochs} EPOCHS:")
            print("-" * 80)
            print(f"{'#':<3} | {'ID':<10} | {'Name':<35} | {'Epochs':<8} | {'Created':<19} | {'Status'}")
            print("-" * 80)
            
            for i, (run, epochs) in enumerate(short_runs, 1):
                created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                status = getattr(run, 'state', 'unknown')
                name = run.name[:32] + '...' if len(run.name) > 35 else run.name
                print(f"{i:<3} | {run.id:<10} | {name:<35} | {epochs:<8} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status}")
        
        if runs_without_epochs:
            print(f"\nRUNS WITHOUT EPOCH DATA:")
            print("-" * 80)
            print(f"{'#':<3} | {'ID':<10} | {'Name':<35} | {'Reason':<20} | {'Created':<19} | {'Status'}")
            print("-" * 80)
            
            for i, (run, reason) in enumerate(runs_without_epochs, 1):
                created_at = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                status = getattr(run, 'state', 'unknown')
                name = run.name[:32] + '...' if len(run.name) > 35 else run.name
                print(f"{i:<3} | {run.id:<10} | {name:<35} | {reason:<20} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<19} | {status}")
        
        # Show some statistics
        if runs_with_epochs:
            epoch_counts = [epochs for _, epochs in runs_with_epochs]
            print(f"\nEPOCH STATISTICS:")
            print(f"- Average epochs: {sum(epoch_counts) / len(epoch_counts):.1f}")
            print(f"- Min epochs: {min(epoch_counts)}")
            print(f"- Max epochs: {max(epoch_counts)}")
            print(f"- Runs with 0 epochs: {sum(1 for e in epoch_counts if e == 0)}")
            print(f"- Runs with 1-4 epochs: {sum(1 for e in epoch_counts if 1 <= e < 5)}")
            print(f"- Runs with 5+ epochs: {sum(1 for e in epoch_counts if e >= 5)}")
        
        return short_runs, runs_without_epochs
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    entity = "anton-safarevich-arc-virtual-cell-challenge"
    project = "state"
    min_epochs = 5
    
    print(f"Finding runs with less than {min_epochs} epochs in {entity}/{project}")
    short_runs, no_epoch_runs = analyze_runs(entity, project, min_epochs)
    
    if short_runs or no_epoch_runs:
        total_candidates = len(short_runs) + len(no_epoch_runs)
        print(f"\nSUMMARY: {total_candidates} runs are candidates for deletion")
        print("(runs with < 5 epochs or no epoch data)")
    else:
        print("\nNo short runs found.")

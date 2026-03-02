#!/usr/bin/env python3
"""
Analyze training convergence and ensemble performance.
Compares individual fold results with ensemble predictions.
"""

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_training_visualization(
    folds: Dict,
    ensemble_analysis: Dict,
    output_dir: Path,
    dataset_name: str,
    timestamp: str,
) -> None:
    """Create bar chart(s): Dice per fold + ensemble; optional training Dice."""
    if not HAS_MATPLOTLIB:
        return
    fold_nums = sorted(folds.keys(), key=lambda x: int(x))
    test_dice = []
    val_dice = []
    train_dice = []
    for fn in fold_nums:
        fd = folds[fn]
        ti = fd.get('training', {})
        vs = fd.get('validation')
        train_dice.append(ti.get('best_dice'))
        val_dice.append(vs.get('mean', {}).get('Dice', {}).get('mean') if vs else None)
        fold_test = None
        if ensemble_analysis and 'folds' in ensemble_analysis:
            fm = ensemble_analysis['folds'].get(fn, {})
            fold_test = fm.get('mean_dice') or fm.get('foreground_dice')
        test_dice.append(fold_test)
    ensemble_dice = None
    if ensemble_analysis and ensemble_analysis.get('ensemble'):
        ensemble_dice = (
            ensemble_analysis['ensemble'].get('mean_dice')
            or ensemble_analysis['ensemble'].get('foreground_dice')
        )
    has_test = any(x is not None for x in test_dice) or (ensemble_dice is not None)
    has_val = any(x is not None for x in val_dice)
    has_train = any(x is not None for x in train_dice)
    if not (has_test or has_val or has_train):
        return
    x_labels = [f'Fold {f}' for f in fold_nums]
    if ensemble_dice is not None:
        x_labels.append('Ensemble')
    x_pos = list(range(len(x_labels)))
    n_series = sum([has_test, has_val, has_train])
    width = 0.25 if n_series >= 2 else 0.55
    fig, ax = plt.subplots(figsize=(max(7, len(x_labels) * 1.2), 5))
    offset = (n_series - 1) * width / 2
    if has_test:
        vals = [(x if x is not None else 0) for x in test_dice]
        if ensemble_dice is not None:
            vals.append(ensemble_dice)
        colors = ['#2E86AB'] * len(fold_nums)
        if ensemble_dice is not None:
            colors.append('#6A994E')
        ax.bar([p - offset for p in x_pos], vals, width=width, label='Test Dice', color=colors, alpha=0.9)
        offset -= width
    if has_val:
        vals = [(x if x is not None else 0) for x in val_dice]
        if ensemble_dice is not None:
            vals.append(0)
        ax.bar([p - offset for p in x_pos], vals, width=width, label='Validation Dice', color='#A23B72', alpha=0.9)
        offset -= width
    if has_train:
        vals = [(x if x is not None else 0) for x in train_dice]
        if ensemble_dice is not None:
            vals.append(0)
        ax.bar([p - offset for p in x_pos], vals, width=width, label='Training Dice (best)', color='#F18F01', alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Dice', fontsize=12)
    ax.set_title(f'Training & Ensemble Analysis — {dataset_name}', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = output_dir / f'training_analysis_{dataset_name}_{timestamp}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Visualization saved to: {out_path}")


def parse_training_log(log_file: Path) -> Dict:
    """Parse training log to extract convergence information."""
    if not log_file.exists():
        return {}
    
    epochs = []
    dice_scores = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch information
    epoch_pattern = r'Epoch (\d+)'
    dice_pattern = r'New best EMA pseudo Dice: ([\d.]+)'
    val_loss_pattern = r'val_loss ([\d.-]+)'
    
    epochs_found = re.findall(epoch_pattern, content)
    dice_found = re.findall(dice_pattern, content)
    val_losses_found = re.findall(val_loss_pattern, content)
    
    return {
        'total_epochs': int(epochs_found[-1]) if epochs_found else None,
        'best_dice': float(dice_found[-1]) if dice_found else None,
        'all_dice': [float(d) for d in dice_found],
        'final_val_loss': float(val_losses_found[-1]) if val_losses_found else None,
        'converged': len(dice_found) > 0 and len(dice_found[-10:]) > 0
    }


def read_summary_json(json_file: Path) -> Optional[Dict]:
    """Read and parse summary.json file."""
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {json_file}: {e}")
        return None


def analyze_training_convergence(results_dir: Path) -> Dict:
    """Analyze training convergence for all folds."""
    folds = {}
    
    for fold_dir in sorted(results_dir.glob('fold_*')):
        if not fold_dir.is_dir():
            continue
        
        fold_num = fold_dir.name.split('_')[1]
        log_files = list(fold_dir.glob('training_log*.txt'))
        
        if not log_files:
            continue
        
        log_file = log_files[0]  # Take first log file
        training_info = parse_training_log(log_file)
        
        # Get validation summary
        val_summary = read_summary_json(fold_dir / 'validation' / 'summary.json')
        
        folds[fold_num] = {
            'training': training_info,
            'validation': val_summary
        }
    
    return folds


def analyze_ensemble_performance(ensemble_dir: Path, individual_folds: Dict, results_dir: Path) -> Dict:
    """Compare ensemble predictions with individual fold predictions."""
    ensemble_summary = read_summary_json(ensemble_dir / 'summary.json')
    
    if not ensemble_summary:
        return {}
    
    # Extract metrics
    ensemble_metrics = {}
    fold_metrics = {}
    
    if ensemble_summary:
        # Try different possible structures
        mean_dice = None
        mean_hd95 = None
        
        # Check foreground_mean first (for test set)
        if 'foreground_mean' in ensemble_summary:
            mean_dice = ensemble_summary['foreground_mean'].get('Dice')
        
        # Check mean structure
        if mean_dice is None and 'mean' in ensemble_summary:
            dice_dict = ensemble_summary['mean'].get('Dice', {})
            if isinstance(dice_dict, dict):
                mean_dice = dice_dict.get('mean')
            elif isinstance(dice_dict, (int, float)):
                mean_dice = dice_dict
        
        ensemble_metrics = {
            'mean_dice': mean_dice,
            'foreground_dice': ensemble_summary.get('foreground_mean', {}).get('Dice'),
        }
    
    # Get test set results for each fold
    predictions_dir = results_dir / 'predictions_imagesTs'
    for fold_num in individual_folds.keys():
        fold_pred_dir = predictions_dir / f'fold_{fold_num}'
        fold_summary = read_summary_json(fold_pred_dir / 'summary.json')
        
        if fold_summary:
            mean_dice = None
            if 'foreground_mean' in fold_summary:
                mean_dice = fold_summary['foreground_mean'].get('Dice')
            elif 'mean' in fold_summary:
                dice_dict = fold_summary['mean'].get('Dice', {})
                if isinstance(dice_dict, dict):
                    mean_dice = dice_dict.get('mean')
                elif isinstance(dice_dict, (int, float)):
                    mean_dice = dice_dict
            
            fold_metrics[fold_num] = {
                'mean_dice': mean_dice,
                'foreground_dice': fold_summary.get('foreground_mean', {}).get('Dice'),
            }
    
    return {
        'ensemble': ensemble_metrics,
        'folds': fold_metrics,
        'improvement': {}
    }


def check_convergence_status(training_info: Dict) -> Tuple[bool, str]:
    """Check if model has converged or could benefit from more epochs."""
    if not training_info.get('all_dice'):
        return False, "No training data available"
    
    dice_history = training_info['all_dice']
    
    if len(dice_history) < 10:
        return False, "Not enough epochs to assess convergence"
    
    # Check last 10 epochs for improvement
    recent_dice = dice_history[-10:]
    improvement = max(recent_dice) - min(recent_dice)
    
    if improvement < 0.001:  # Less than 0.1% improvement
        return True, f"Converged (improvement < 0.001 in last 10 epochs)"
    elif improvement < 0.005:  # Less than 0.5% improvement
        return True, f"Likely converged (improvement < 0.005 in last 10 epochs)"
    else:
        return False, f"Still improving (improvement: {improvement:.4f} in last 10 epochs)"


def print_analysis(folds: Dict, ensemble_analysis: Dict):
    """Print formatted analysis results."""
    print("\n" + "="*80)
    print("TRAINING CONVERGENCE ANALYSIS")
    print("="*80)
    
    for fold_num in sorted(folds.keys()):
        fold_data = folds[fold_num]
        training_info = fold_data.get('training', {})
        val_summary = fold_data.get('validation')
        
        print(f"\n--- Fold {fold_num} ---")
        
        if training_info:
            total_epochs = training_info.get('total_epochs')
            best_dice = training_info.get('best_dice')
            final_val_loss = training_info.get('final_val_loss')
            
            print(f"  Total Epochs: {total_epochs}")
            print(f"  Best Training Dice: {best_dice:.4f}" if best_dice else "  Best Training Dice: N/A")
            print(f"  Final Validation Loss: {final_val_loss:.4f}" if final_val_loss else "  Final Validation Loss: N/A")
            
            converged, status = check_convergence_status(training_info)
            print(f"  Convergence Status: {status}")
            if not converged:
                print(f"  → RECOMMENDATION: More epochs might help")
            else:
                print(f"  → RECOMMENDATION: Model has converged")
        
        if val_summary:
            mean_dice = val_summary.get('mean', {}).get('Dice', {}).get('mean')
            mean_hd95 = val_summary.get('mean', {}).get('95th_percentile_of_HD', {}).get('mean')
            print(f"  Validation Dice: {mean_dice:.4f}" if mean_dice else "  Validation Dice: N/A")
            print(f"  Validation HD95: {mean_hd95:.2f}mm" if mean_hd95 else "  Validation HD95: N/A")
    
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE ANALYSIS")
    print("="*80)
    
    if ensemble_analysis:
        ensemble_metrics = ensemble_analysis.get('ensemble', {})
        fold_metrics = ensemble_analysis.get('folds', {})
        
        print(f"\nEnsemble Metrics:")
        if ensemble_metrics.get('mean_dice'):
            print(f"  Mean Dice: {ensemble_metrics['mean_dice']:.4f}")
        if ensemble_metrics.get('mean_hd95'):
            print(f"  Mean HD95: {ensemble_metrics['mean_hd95']:.2f}mm")
        
        print(f"\nIndividual Fold Metrics (Test Set):")
        for fold_num in sorted(fold_metrics.keys()):
            metrics = fold_metrics[fold_num]
            print(f"  Fold {fold_num}:")
            if metrics.get('mean_dice'):
                print(f"    Dice: {metrics['mean_dice']:.4f}")
            elif metrics.get('foreground_dice'):
                print(f"    Dice (foreground): {metrics['foreground_dice']:.4f}")
            if metrics.get('mean_hd95'):
                print(f"    HD95: {metrics['mean_hd95']:.2f}mm")
        
        # Calculate improvement
        dice_to_use = ensemble_metrics.get('mean_dice') or ensemble_metrics.get('foreground_dice')
        if dice_to_use and fold_metrics:
            fold_dice_values = []
            for m in fold_metrics.values():
                dice_val = m.get('mean_dice') or m.get('foreground_dice')
                if dice_val:
                    fold_dice_values.append(dice_val)
            
            if fold_dice_values:
                avg_fold_dice = sum(fold_dice_values) / len(fold_dice_values)
                ensemble_dice = dice_to_use
                improvement = ensemble_dice - avg_fold_dice
                improvement_pct = (improvement / avg_fold_dice) * 100 if avg_fold_dice > 0 else 0
                
                print(f"\nEnsemble vs Average Fold Performance (Test Set):")
                print(f"  Average Fold Dice: {avg_fold_dice:.4f}")
                print(f"  Ensemble Dice: {ensemble_dice:.4f}")
                print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
                
                if improvement > 0.01:
                    print(f"  → Ensemble provides SIGNIFICANT improvement!")
                elif improvement > 0.005:
                    print(f"  → Ensemble provides moderate improvement")
                elif improvement > 0:
                    print(f"  → Ensemble provides slight improvement")
                else:
                    print(f"  → Ensemble does not improve over average fold")
    else:
        print("\n[WARN] No ensemble analysis available. Run ensemble predictions first.")


def save_analysis_results(folds: Dict, ensemble_analysis: Dict, output_dir: Path, dataset_name: str):
    """Save analysis results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for CSV
    csv_data = []
    
    # Training convergence data
    for fold_num in sorted(folds.keys()):
        fold_data = folds[fold_num]
        training_info = fold_data.get('training', {})
        val_summary = fold_data.get('validation')
        
        row = {
            'fold': fold_num,
            'total_epochs': training_info.get('total_epochs', ''),
            'best_training_dice': training_info.get('best_dice', ''),
            'final_val_loss': training_info.get('final_val_loss', ''),
            'converged': check_convergence_status(training_info)[0],
            'convergence_status': check_convergence_status(training_info)[1],
        }
        
        # Add validation metrics if available
        if val_summary:
            row['validation_dice'] = val_summary.get('mean', {}).get('Dice', {}).get('mean', '')
            row['validation_hd95'] = val_summary.get('mean', {}).get('95th_percentile_of_HD', {}).get('mean', '')
        else:
            row['validation_dice'] = ''
            row['validation_hd95'] = ''
        
        # Add test set metrics if available
        if ensemble_analysis and 'folds' in ensemble_analysis:
            fold_metrics = ensemble_analysis['folds'].get(fold_num, {})
            row['test_dice'] = fold_metrics.get('mean_dice') or fold_metrics.get('foreground_dice', '')
        else:
            row['test_dice'] = ''
        
        csv_data.append(row)
    
    # Add ensemble row
    if ensemble_analysis and ensemble_analysis.get('ensemble'):
        ensemble_metrics = ensemble_analysis['ensemble']
        ensemble_row = {
            'fold': 'ensemble',
            'total_epochs': '',
            'best_training_dice': '',
            'final_val_loss': '',
            'converged': '',
            'convergence_status': '',
            'validation_dice': '',
            'validation_hd95': '',
            'test_dice': ensemble_metrics.get('mean_dice') or ensemble_metrics.get('foreground_dice', ''),
        }
        csv_data.append(ensemble_row)
    
    # Save CSV
    csv_file = output_dir / f'training_analysis_{dataset_name}_{timestamp}.csv'
    if csv_data:
        fieldnames = ['fold', 'total_epochs', 'best_training_dice', 'final_val_loss', 
                     'converged', 'convergence_status', 'validation_dice', 'validation_hd95', 'test_dice']
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n[INFO] CSV results saved to: {csv_file}")
    
    # Save detailed JSON
    json_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'folds': {},
        'ensemble': ensemble_analysis.get('ensemble', {}) if ensemble_analysis else {},
        'summary': {}
    }
    
    for fold_num, fold_data in folds.items():
        json_data['folds'][fold_num] = {
            'training': fold_data.get('training', {}),
            'validation': fold_data.get('validation', {}),
            'test': ensemble_analysis.get('folds', {}).get(fold_num, {}) if ensemble_analysis else {}
        }
    
    # Add summary statistics
    if ensemble_analysis and ensemble_analysis.get('ensemble') and ensemble_analysis.get('folds'):
        fold_dice_values = []
        for m in ensemble_analysis['folds'].values():
            dice_val = m.get('mean_dice') or m.get('foreground_dice')
            if dice_val:
                fold_dice_values.append(dice_val)
        
        if fold_dice_values:
            ensemble_dice = ensemble_analysis['ensemble'].get('mean_dice') or ensemble_analysis['ensemble'].get('foreground_dice')
            if ensemble_dice:
                avg_fold_dice = sum(fold_dice_values) / len(fold_dice_values)
                improvement = ensemble_dice - avg_fold_dice
                improvement_pct = (improvement / avg_fold_dice) * 100 if avg_fold_dice > 0 else 0
                
                json_data['summary'] = {
                    'average_fold_dice': avg_fold_dice,
                    'ensemble_dice': ensemble_dice,
                    'improvement': improvement,
                    'improvement_percent': improvement_pct
                }
    
    json_file = output_dir / f'training_analysis_{dataset_name}_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[INFO] JSON results saved to: {json_file}")

    create_training_visualization(folds, ensemble_analysis, output_dir, dataset_name, timestamp)


def main():
    parser = argparse.ArgumentParser(description='Analyze training convergence and ensemble performance')
    parser.add_argument('--dataset', type=str, default='Dataset001_GroundTruth',
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='3d_fullres',
                       help='Configuration name')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer',
                       help='Trainer name')
    parser.add_argument('--plans', type=str, default='nnUNetPlans',
                       help='Plans name')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for CSV/JSON files (default: analysis_results/training/)')
    
    args = parser.parse_args()
    
    results_dir = PROJECT_ROOT / 'data' / 'nnUNet_results' / args.dataset / f'{args.trainer}__{args.plans}__{args.config}'
    ensemble_dir = PROJECT_ROOT / 'ensemble_predictions' / f'{args.dataset}_{args.config}'
    
    print(f"Analyzing results in: {results_dir}")
    print(f"Ensemble directory: {ensemble_dir}")
    
    # Analyze training convergence
    folds = analyze_training_convergence(results_dir)
    
    # Analyze ensemble performance
    ensemble_analysis = analyze_ensemble_performance(ensemble_dir, folds, results_dir)
    
    # Print results
    print_analysis(folds, ensemble_analysis)
    
    # Save results to CSV and JSON
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'analysis_results' / 'training'
    save_analysis_results(folds, ensemble_analysis, output_dir, args.dataset)


if __name__ == '__main__':
    main()


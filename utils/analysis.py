"""
Analysis and scoring utilities for the Iowa Gambling Task.

Provides functions for computing IGT metrics, block-wise analysis,
and learning curve visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from utils.igt_task import TrialResult


@dataclass
class BlockMetrics:
    """Metrics for a block of trials."""
    block: int
    start_trial: int
    end_trial: int
    net_score: int
    advantageous_choices: int
    total_choices: int
    proportion_advantageous: float
    deck_counts: Dict[str, int]
    total_money: int


def compute_block_metrics(history: List[TrialResult], block_size: int = 20) -> List[BlockMetrics]:
    """
    Compute block-wise metrics for IGT performance.
    
    Args:
        history: List of trial results
        block_size: Number of trials per block (default: 20)
        
    Returns:
        List of BlockMetrics for each block
    """
    if not history:
        return []
    
    n_trials = len(history)
    n_blocks = (n_trials + block_size - 1) // block_size  # Ceiling division
    
    block_metrics = []
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_trials)
        block_trials = history[start_idx:end_idx]
        
        # Count deck selections in this block
        deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for trial in block_trials:
            deck_counts[trial.deck] += 1
        advantageous_decks = trial.adv_decks.split(',')
        
        # Compute metrics
        advantageous_choices = sum(
            deck_counts[deck] for deck in advantageous_decks)
        total_choices = len(block_trials)
        disadvantageous_choices = total_choices - advantageous_choices
        net_score = advantageous_choices - disadvantageous_choices
        proportion_advantageous = advantageous_choices / total_choices if total_choices > 0 else 0.0
        
        # Total money at end of block
        total_money = block_trials[-1].cumulative if block_trials else 0
        
        block_metrics.append(BlockMetrics(
            block=block_idx + 1,
            start_trial=start_idx,
            end_trial=end_idx - 1,
            net_score=net_score,
            advantageous_choices=advantageous_choices,
            total_choices=total_choices,
            proportion_advantageous=proportion_advantageous,
            deck_counts=deck_counts,
            total_money=total_money
        ))
    
    return block_metrics


def compute_summary_metrics(history: List[TrialResult]) -> Dict:
    """
    Compute summary metrics for entire IGT session.
    
    Args:
        history: List of trial results
        
    Returns:
        Dictionary with summary metrics
    """
    if not history:
        return {
            'n_trials': 0,
            'net_score': 0,
            'total_money': 0,
            'deck_counts': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
            'proportion_advantageous': 0.0
        }
    
    # Count deck selections
    deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for trial in history:
        deck_counts[trial.deck] += 1
    
    # Compute metrics
    adv_decks = history[0].adv_decks.split(',')
    advantageous_choices = sum(deck_counts[deck] for deck in adv_decks)
    net_score = advantageous_choices - \
        (len(history) - advantageous_choices)
    # (deck_counts['C'] + deck_counts['D']) - (deck_counts['A'] + deck_counts['B'])
    total_money = history[-1].cumulative
    
    return {
        'n_trials': len(history),
        'net_score': net_score,
        'total_money': total_money,
        'starting_money': history[0].cumulative - history[0].net if history else 0,
        'net_gain': total_money - (history[0].cumulative - history[0].net) if history else 0,
        'deck_counts': deck_counts,
        'proportion_advantageous': advantageous_choices / len(history)
    }


def plot_learning_curve(history: List[TrialResult], block_size: int = 20, 
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curve showing shift from disadvantageous to advantageous decks.
    
    Args:
        history: List of trial results
        block_size: Number of trials per block (default: 20)
        save_path: Path to save figure (if None, displays instead)
    """
    adv_decks = history[0].adv_decks.split(',')
    block_metrics = compute_block_metrics(history, block_size)
    
    if not block_metrics:
        print("No data to plot")
        return
    
    blocks = [m.block for m in block_metrics]
    proportions = [m.proportion_advantageous for m in block_metrics]
    net_scores = [m.net_score for m in block_metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot proportion of advantageous choices
    ax1.plot(blocks, proportions, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    ax1.set_xlabel('Block')
    ax1.set_ylabel('Proportion Advantageous Choices')
    ax1.set_title(
        f'Learning Curve: Proportion of Advantageous Choices ({adv_decks}) per Block')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot net score per block
    ax2.plot(blocks, net_scores, 's-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', label='Neutral')
    ax2.set_xlabel('Block')
    ax2.set_ylabel(f'Net Score (Adv {adv_decks}) - (Disadv)')
    ax2.set_title('Net Score per Block')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_deck_choices(history: List[TrialResult], save_path: Optional[str] = None) -> None:
    """
    Plot deck choice distribution over time.
    
    Args:
        history: List of trial results
        save_path: Path to save figure (if None, displays instead)
    """
    if not history:
        print("No data to plot")
        return
    
    # Extract trials and deck choices
    trials = [t.trial for t in history]
    decks = [t.deck for t in history]
    
    # Convert to numeric for plotting
    adv_decks = history[0].adv_decks.split(',')
    disadv_decks = [d for d in ['A', 'B', 'C', 'D'] if d not in adv_decks]
    sorted_decks = disadv_decks + adv_decks  # Disadvantaged first
    deck_to_num = {deck: idx for idx, deck in enumerate(sorted_decks)}
    deck_nums = [deck_to_num[d] for d in decks]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot choice sequence
    colors = {'A': 'red', 'B': 'orange', 'C': 'lightblue', 'D': 'blue'}
    for trial, deck in zip(trials, decks):
        ax1.scatter(trial, deck_to_num[deck], c=colors[deck], s=50, alpha=0.6)
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Deck')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels([f'{deck} (Bad)' if deck in disadv_decks else f'{deck} (Good)' for deck in sorted_decks])
    ax1.set_title('Deck Choices Over Time')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axhline(y=1.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Plot cumulative deck counts
    cumulative_counts = {'A': [], 'B': [], 'C': [], 'D': []}
    current_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for trial in history:
        current_counts[trial.deck] += 1
        for deck in ['A', 'B', 'C', 'D']:
            cumulative_counts[deck].append(current_counts[deck])
    
    for deck in ['A', 'B', 'C', 'D']:
        ax2.plot(trials, cumulative_counts[deck], label=f'Deck {deck}', 
                linewidth=2, color=colors[deck])
    
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_title('Cumulative Deck Selections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_report(history: List[TrialResult], block_size: int = 20) -> None:
    """
    Print a comprehensive summary report of IGT performance.
    
    Args:
        history: List of trial results
        block_size: Number of trials per block (default: 20)
    """
    print("\n" + "="*60)
    print("IOWA GAMBLING TASK - SUMMARY REPORT")
    print("="*60)
    
    # Overall summary
    summary = compute_summary_metrics(history)
    print(f"\nTotal Trials: {summary['n_trials']}")
    print(f"Starting Money: ${summary['starting_money']}")
    print(f"Final Money: ${summary['total_money']}")
    print(f"Net Gain/Loss: ${summary['net_gain']}")
    
    print(f"\nDeck Selection Counts:")
    for deck in ['A', 'B', 'C', 'D']:
        count = summary['deck_counts'][deck]
        pct = (count / summary['n_trials'] * 100) if summary['n_trials'] > 0 else 0
        print(f"  Deck {deck}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nIGT Net Score (C+D) - (A+B): {summary['net_score']}")
    print(f"Proportion Advantageous Choices: {summary['proportion_advantageous']:.3f}")
    
    # Block-wise analysis
    print(f"\n{'='*60}")
    print("BLOCK-WISE ANALYSIS")
    print(f"{'='*60}")
    
    block_metrics = compute_block_metrics(history, block_size)
    
    print(f"\n{'Block':<8} {'Trials':<12} {'Net Score':<12} {'Prop. Adv.':<12} {'Total $':<12}")
    print("-" * 60)
    
    for bm in block_metrics:
        trial_range = f"{bm.start_trial}-{bm.end_trial}"
        print(f"{bm.block:<8} {trial_range:<12} {bm.net_score:<12} "
              f"{bm.proportion_advantageous:<12.3f} ${bm.total_money:<11}")
    
    print("\n" + "="*60)

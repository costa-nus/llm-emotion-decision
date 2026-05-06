"""
Iowa Gambling Task (IGT) Environment Module

This module implements the classic Iowa Gambling Task, a psychological paradigm
used to study decision-making under uncertainty and learning from rewards and punishments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .agents import MaxLimitReachedError


@dataclass
class TrialResult:
    """Result of a single trial in the IGT."""
    trial: int
    deck: str
    reward: int
    loss: int
    net: int
    cumulative: int
    adv_decks: str


class IGTEnvironment:
    """
    Iowa Gambling Task Environment.
    
    Implements four decks (A, B, C, D) with standard IGT payoff structures:
    - Decks A and B: High immediate rewards ($100), but occasional large losses (disadvantageous)
    - Decks C and D: Lower immediate rewards ($50), but smaller losses (advantageous)
    
    Deck A: $100 reward, $250 loss every 10 trials on average → net -$25/trial
    Deck B: $100 reward, $1250 loss once every 10 trials → net -$25/trial
    Deck C: $50 reward, $50 loss every 10 trials on average → net +$25/trial
    Deck D: $50 reward, $250 loss once every 10 trials → net +$25/trial
    """
    
    # Standard IGT payoff structure based on Bechara et al. (1994)
    DECK_CONFIGS = {
        "A": {
            "reward": 100,
            # Frequent, irregular moderate losses (40 cards total)
            "loss_schedule": [
                0, 0, -150, 0, -300, 0, -200, 0, -250, -350,
                0, -350, 0, -250, -200, 0, -300, -150, 0, 0,
                0, -300, 0, -350, 0, -200, -250, -150, 0, 0,
                -350, -200, -250, 0, 0, 0, -150, -300, 0, 0,
            ],
            # (40*100 - sum(losses)) / 40 = -25
            "expected_value": -25
        },

        "B": {
            "reward": 100,
            # Rare but catastrophic losses
            "loss_schedule": [
                0, 0, 0, 0, 0, 0, 0, 0, -1250, 0,
                0, 0, 0, -1250, 0, 0, 0, 0, 0, 0,
                -1250, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, -1250, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            # (40*100 - 4*1250) / 40 = -25
            "expected_value": -25
        },

        "C": {
            "reward": 50,
            # Frequent small losses, irregular placement
            "loss_schedule": [
                0, 0, -50, 0, -50, 0, -50, 0, -50, -50,
                0, -25, -75, 0, 0, 0, -25, -75, 0, -50,
                0, 0, 0, -50, -25, -50, 0, 0, -75, -50,
                0, 0, 0, -25, -25, 0, -75, 0, -50, -75,
            ],
            # (40*50 - sum(losses)) / 40 = +25
            "expected_value": 25
        },

        "D": {
            "reward": 50,
            # Rare moderate losses
            "loss_schedule": [
                0, 0, 0, 0, 0, 0, 0, 0, 0, -250,
                0, 0, 0, 0, 0, 0, 0, 0, 0, -250,
                0, 0, 0, 0, 0, 0, 0, 0, -250, 0,
                0, 0, 0, 0, -250, 0, 0, 0, 0, 0,
            ],
            # (40*50 - 4*250) / 40 = +25
            "expected_value": 25
        }
    }
    
    def __init__(self, n_trials: int = 100, starting_money: int = 2000, seed: Optional[int] = None):
        """
        Initialize the IGT environment.
        
        Args:
            n_trials: Number of trials in the experiment (default: 100)
            starting_money: Starting amount of money (default: 2000)
            seed: Random seed for reproducibility (default: None)
        """
        self.n_trials = n_trials
        self.starting_money = starting_money
        self.seed = seed
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
        # Game state
        self.current_trial = 0
        self.cumulative_money = starting_money
        self.history: List[TrialResult] = []
        
        # Deck counters (track position in loss schedule)
        self.deck_counters = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        # Track which decks are advantaged vs disadvantaged
        self.advantaged_decks = {'C', 'D'}  # Decks with positive expected value
        self.disadvantaged_decks = {'A', 'B'}  # Decks with negative expected value
        
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.current_trial = 0
        self.cumulative_money = self.starting_money
        self.history = []
        self.deck_counters = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.advantaged_decks = {'C', 'D'}
        self.disadvantaged_decks = {'A', 'B'}
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)
    
    def step(self, deck: str) -> TrialResult:
        """
        Execute one trial: player chooses a deck and receives outcome.
        
        Args:
            deck: Deck choice ('A', 'B', 'C', or 'D')
            
        Returns:
            TrialResult with trial details
            
        Raises:
            ValueError: If deck is invalid or game is over
        """
        if deck not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid deck: {deck}. Must be one of ['A', 'B', 'C', 'D']")
        
        if self.current_trial >= self.n_trials:
            raise ValueError("Game is over. No more trials allowed.")
        
        # Get reward for this deck
        reward = self.DECK_CONFIGS[deck]['reward']
        
        # Get loss based on schedule (cycles through the schedule)
        loss_schedule = self.DECK_CONFIGS[deck]['loss_schedule']
        loss = loss_schedule[self.deck_counters[deck] % len(loss_schedule)]
        
        # Update deck counter
        self.deck_counters[deck] += 1
        
        # Calculate net outcome
        net = reward + loss  # loss is already negative
        
        # Update cumulative money
        self.cumulative_money += net
        
        # Record trial
        trial_result = TrialResult(
            trial=self.current_trial,
            deck=deck,
            reward=reward,
            loss=loss,
            net=net,
            cumulative=self.cumulative_money,
            adv_decks=','.join(sorted(self.advantaged_decks))
        )
        
        self.history.append(trial_result)
        self.current_trial += 1
        
        return trial_result
    
    def is_done(self) -> bool:
        """Check if all trials are completed."""
        return self.current_trial >= self.n_trials
    
    def get_history(self) -> List[TrialResult]:
        """Get complete trial history."""
        return self.history.copy()
    
    def get_deck_counts(self) -> Dict[str, int]:
        """Get number of times each deck was chosen."""
        counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for trial in self.history:
            counts[trial.deck] += 1
        return counts
    
    def shuffle_payoffs(self, seed: Optional[int] = None, periodic: bool = False) -> None:
        """
        Shuffle the reward/loss configuration among decks without changing deck names.
        
        This creates different advantage structures for the decks while maintaining
        the deck labels (A, B, C, D). The advantaged/disadvantaged groups are tracked
        so that net score always reflects: (advantaged decks) - (disadvantaged decks).
        
        Args:
            seed: Random seed for reproducibility (default: None)
            periodic: Whether to use a periodic shuffle (default: False)
        """
        # Get the payoff structures (values only, not keys)
        payoff_structures = list(self.DECK_CONFIGS.values())
        if periodic:
            if seed is None:
                seed = self.seed
            # adv_decks = [2, 3]  # Indices of advantaged decks
            # disadv_decks = [0, 1]  # Indices of disadvantaged decks
            candidate_adv_decks = [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 3],
            ]
            # Select a random candidate for the new advantaged decks
            selected_adv_decks = candidate_adv_decks[seed % len(candidate_adv_decks)]
            # Update adv/disadv decks based on selected indices
            disadv_decks = [i for i in range(4) if i not in selected_adv_decks]
            
            new_payoff_structures = [None] * 4
            new_payoff_structures[selected_adv_decks[0]] = payoff_structures[2]
            new_payoff_structures[selected_adv_decks[1]] = payoff_structures[3]
            new_payoff_structures[disadv_decks[0]] = payoff_structures[0]
            new_payoff_structures[disadv_decks[1]] = payoff_structures[1]
            payoff_structures = new_payoff_structures
        else:
            rng = self.rng if seed is None else np.random.RandomState(seed)
            
            # Shuffle the structures
            rng.shuffle(payoff_structures)
        
        # Reassign shuffled structures to deck names
        deck_names = ['A', 'B', 'C', 'D']
        for deck_name, config in zip(deck_names, payoff_structures):
            self.DECK_CONFIGS[deck_name] = config
        
        # Determine which decks are now advantaged (expected_value > 0)
        self.advantaged_decks = {deck for deck, config in self.DECK_CONFIGS.items() 
                                 if config['expected_value'] > 0}
        self.disadvantaged_decks = {deck for deck, config in self.DECK_CONFIGS.items() 
                                    if config['expected_value'] <= 0}
        
        # Reset deck counters since structure changed
        self.deck_counters = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        print(f"** Payoffs shuffled. New advantaged decks: {self.advantaged_decks}, disadvantaged decks: {self.disadvantaged_decks}")
    
    def get_net_score(self) -> int:
        """
        Calculate IGT Net Score: (advantaged decks) - (disadvantaged decks).
        
        Positive score indicates advantageous decision-making.
        The advantage/disadvantage is determined by expected value,
        which can change if payoffs have been shuffled.
        """
        counts = self.get_deck_counts()
        advantaged_count = sum(counts[deck] for deck in self.advantaged_decks)
        disadvantaged_count = sum(counts[deck] for deck in self.disadvantaged_decks)
        return advantaged_count - disadvantaged_count
    
    def get_total_money(self) -> int:
        """Get total money accumulated."""
        return self.cumulative_money


def run_igt_simulation(env: IGTEnvironment, agent, verbose: bool = False, shuffle_payoffs=False, periodic_shuffle=False) -> list:
    """
    Run a complete IGT simulation.
    
    Args:
        env: IGT environment
        agent: Agent to play the task
        verbose: Whether to print trial-by-trial information
        
    Returns:
        List of trial results
    """
    env.reset()
    if shuffle_payoffs:
        assert not periodic_shuffle, "Cannot use both shuffle_payoffs and periodic_shuffle simultaneously."
        env.shuffle_payoffs()
    elif periodic_shuffle:
        assert not shuffle_payoffs, "Cannot use both shuffle_payoffs and periodic_shuffle simultaneously."
        env.shuffle_payoffs(seed=env.seed, periodic=True)
    agent.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting IGT Simulation with {agent.__class__.__name__}")
        print(f"{'='*60}\n")

    try:
        while not env.is_done():
            # Agent chooses deck
            deck = agent.choose_deck()

            # Execute trial
            result = env.step(deck)

            # Agent learns from outcome
            agent.update(result.deck, result.reward, result.loss, result.net)

            if verbose:
                print(f"Trial {result.trial+1:3d}: Deck {result.deck} → "
                    f"Reward = {result.reward:4d}, Loss = {result.loss:5d}, "
                    f"Net = {result.net:4d}, Total = {result.cumulative:5d}")
    except MaxLimitReachedError as e:
        print(f"Error during simulation: MaxLimitReachedError - {e}")
    except Exception as e:
        print(f"Unknown Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    if verbose:
        print(f"{'='*60}\n")

    return env.get_history()

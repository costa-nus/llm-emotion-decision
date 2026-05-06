#!/usr/bin/env python3
"""
Iowa Gambling Task - Complete Demonstration

This script demonstrates the full functionality of the IGT module:
1. Running the IGT with different agents
2. Computing summary metrics and block-wise analysis
3. Visualizing learning curves
4. Fitting reinforcement learning models to behavior

Usage examples:

    python run_igt_llm.py --model hf/Qwen/Qwen2.5-7B-Instruct --n-trials=40 --emotion=anxiety
    python run_igt_llm.py --model hf/openai/gpt-oss-20b --n-trials=40 --emotion=anxiety
    python run_igt_llm.py --model hf/openai/gpt-oss-20b --n-trials=80 --agent=reflexion
    python run_igt_llm.py --model hf/Qwen/Qwen3-4B-Thinking-2507 --n-trials=40
    python run_igt_llm.py --model hf/openai/gpt-oss-20b --n-trials=40 --reasoning_effort=High
"""

import numpy as np
import sys
import os
import argparse
import json
from dataclasses import asdict

# Import IGT modules
from utils.hf import is_thinking_model
from utils.igt_task import IGTEnvironment
from utils.agents import ReActAgent, QueryAgent, ReflexionAgent
from utils.analysis import print_summary_report
from utils.igt_task import run_igt_simulation

import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file


def save_history_to_jsonl(history: list, filepath: str) -> None:
    """
    Save trial history to JSONL file.
    
    Args:
        history: List of TrialResult objects
        filepath: Path to save JSONL file
    """
    with open(filepath, 'w') as f:
        for trial in history:
            trial_dict = asdict(trial)
            f.write(json.dumps(trial_dict) + '\n')
    print(f"History saved to {filepath} ({len(history)} trials)")


def run_llm_agent_igt(agent_type, emotion="none", strength="high",
                      game_prompt="choose", n_trials=100, seed=42, 
                      lm=None, jsonl_filepath=None, gpt_oss=False,
                      shuffle_payoffs=False, periodic_shuffle=False, memory_size=3):
    """Demonstrate IGT with OpenAI agent."""
    print("\n" + "#"*60)
    print(f"# Run Agent (Emotion: {emotion}, Strength: {strength})")
    print("#"*60)

    try:
        # Create environment and agent
        env = IGTEnvironment(n_trials=n_trials, seed=seed)

        if agent_type == 'react':
            assert gpt_oss, "Conversation agent requires gpt-oss or thinking model."
            agent = ReActAgent(lm, emotion=emotion, strength=strength, game_prompt=game_prompt)
        elif agent_type == 'query':
            agent = QueryAgent(lm, emotion=emotion, strength=strength, game_prompt=game_prompt)
        elif agent_type == 'reflexion':
            agent = ReflexionAgent(lm, emotion=emotion, strength=strength, game_prompt=game_prompt, memory_size=memory_size)
        else:
            raise RuntimeError(f"Unknown agent type: {agent_type}")

        # Run simulation
        history = run_igt_simulation(
            env, agent, verbose=True, shuffle_payoffs=shuffle_payoffs,
            periodic_shuffle=periodic_shuffle)

        # Save history to JSONL
        if jsonl_filepath:
            save_history_to_jsonl(history, jsonl_filepath)
            agent.save_to(jsonl_filepath.replace('.jsonl', '_agent_messages.jsonl'))

        # Print summary
        print_summary_report(history, block_size=20)

        return history
    except ImportError as e:
        print(f"\nOpenAI agent demo skipped: {e}")
        print("To use OpenAI agent, install: pip install openai")
        print("And set OPENAI_API_KEY environment variable")
        return None


def main():
    """Run LLM agent demonstrations."""
    parser = argparse.ArgumentParser(
        description="Iowa Gambling Task - LLM Agent Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run_igt_llm.py                    # Run with default model
        python run_igt_llm.py --model openai/gpt-4-turbo  # Use specific model
        python run_igt_llm.py --n-trials 200    # Run 200 trials
                """
    )
    parser.add_argument(
        '--agent',
        type=str, default='conv', choices=['react', 'query', 'reflexion'],
        help='agent type'
    )
    parser.add_argument(
        '--model',
        type=str, default='openai/gpt-4o-mini',
        help='Model for AI agent (default: openai/gpt-4o-mini)'
    )
    parser.add_argument(
        '--emotion',
        type=str, default=None,
        choices=['anxiety', 'sadness', 'anger', 'happiness'],
        help='Emotion for agent (none for no emotion, default: None)'
    )
    parser.add_argument(
        '--strength',
        type=str, default='high',
        choices=['low', 'medium', 'high'],
        help='Emotion strength level (default: high)'
    )
    parser.add_argument(
        '--game_prompt',
        type=str, default='choose',
        choices=['choose','economics','medicine'],
        help='System prompt for AI agent (default: choose)'
    )
    parser.add_argument(
        '--n-trials',
        type=int, default=100,
        help='Number of trials for each demo (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        "--model_server",
        type=str,
        default=None,
        help="Base URL for the model server (if applicable)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='out',
        help="Output directory for saving experiment results (default: out/)"
    )
    parser.add_argument(
        "--jsonl_file",
        type=str,
        default=None,
        help="Output JSONL file to save experiment history (default: None)"
    )
    parser.add_argument(
        "--cache",
        action='store_true',
        help="Enable caching for model responses (default: False)"
    )
    parser.add_argument(
        "--shuffle_payoffs",
        action='store_true',
        help="Shuffle deck payoffs for each new environment (default: False)"
    )
    parser.add_argument(
        "--periodic_shuffle",
        action='store_true',
        help="Periodically shuffle deck payoffs during the task (default: False)"
    )
    parser.add_argument(
        '--reasoning_effort',
        type=str,
        default='Low',
        choices=['No', 'Low', 'Medium', 'High'],
        help='Reasoning effort level for gpt-oss models (default: low)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM sampling (default: 0.7)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p for LLM sampling (default: 0.9)'
    )
    parser.add_argument(
        '--memory_size',
        type=int,
        default=3,
        help='Memory size for reflexion agent (default: 3)'
    )

    args = parser.parse_args()
    if args.emotion == 'none':
        args.emotion = None

    print("\n" + "="*60)
    print(" "*10 + "IOWA GAMBLING TASK - LLM AGENT")
    print("="*60)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"\nOutput directory: {args.output_dir}\n")
    
    # Create base folder for JSONL file if specified
    if not args.jsonl_file:
        args.jsonl_file = os.path.join(
            args.output_dir, 
            'history', 
            f'{args.agent}_{args.model.replace("/", "--")}_{args.emotion}-{args.strength}_s{args.seed}.jsonl')
    jsonl_dir = os.path.dirname(args.jsonl_file)
    if jsonl_dir and not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)

    try:
        if args.model.startswith("hf/"):
            from transformers import set_seed
            from utils.hf import load_chat_lm
            set_seed(args.seed)
            lm = load_chat_lm(args.model[3:],
                              max_new_tokens=32768,
                              do_sample=args.temperature > 0,
                              temperature=args.temperature,
                              top_p=args.top_p,
                              reasoning_effort=args.reasoning_effort)
        elif args.model.startswith("openai/") and 'gpt-5' in args.model:
            from utils.openai import load_openai_lm
            lm = load_openai_lm(
                args.model.split('/')[1],
                max_output_tokens=32768,
                temperature=args.temperature,
                top_p=args.top_p,
                reasoning_effort=args.reasoning_effort
                )

        # Run Agent with specified emotion and strength
        history = run_llm_agent_igt(args.agent, emotion=args.emotion,strength=args.strength, 
                                    game_prompt=args.game_prompt, n_trials=args.n_trials, seed=args.seed,
                                    lm=lm, jsonl_filepath=args.jsonl_file,
                                    gpt_oss=("openai/gpt-oss" in args.model) or is_thinking_model(args.model),
                                    shuffle_payoffs=args.shuffle_payoffs,
                                    periodic_shuffle=args.periodic_shuffle, 
                                    memory_size=args.memory_size)

        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
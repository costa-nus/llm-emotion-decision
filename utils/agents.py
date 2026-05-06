"""
Agent interface and implementations for the Iowa Gambling Task.
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
from typing import Literal, List, Dict
import json
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class Agent(ABC):
    """Base class for IGT agents."""
    
    @abstractmethod
    def choose_deck(self) -> str:
        """
        Choose a deck to play.
        
        Returns:
            Deck choice ('A', 'B', 'C', or 'D')
        """
        pass
    
    @abstractmethod
    def update(self, deck: str, reward: int, loss: int, net: int) -> None:
        """
        Update agent's internal state based on experienced outcome.
        
        Args:
            deck: Deck that was chosen
            reward: Reward received
            loss: Loss incurred (negative value)
            net: Net outcome (reward + loss)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent to initial state."""
        pass

def extract_between_tags(text: str, tag: str) -> str:
    """Extract content between specified XML-like tags."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.rfind(start_tag)
    end_idx = text.rfind(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx + len(start_tag):end_idx].strip()
    elif sum(x in text for x in "ABCD") == 1:
        return next(x for x in "ABCD" if x in text)
    return ""


class MaxLimitReachedError(Exception):
    """Custom exception for exceeding maximum retry attempts."""
    pass

class ChatAgent(Agent):
    """
    Base class for chat-based agents using language models.
    """
    def __init__(self, lm, emotion: str | None = None, strength: Literal["low", "medium", "high"] = "high",
                 game_prompt="choose", max_retry=3):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package is required. Install it with: pip install openai"
            )
            
        self.lm = lm
        self.messages = []
        self._raw_messages = []
        
        self.emotion = emotion
        self.strength = strength
        self.decks = ['A', 'B', 'C', 'D']
        
        self.outcomes = []
        self.total_score = 0
        self.deck_stats = {'A': [], 'B': [], 'C': [], 'D': []}
        self.game_prompt, self.reward_tag, self.loss_tag = self._init_game_prompt(game_prompt)
        self.persona = self._init_persona()
        
        self.add_message({"role": "system", "content": "You will reason and act in response to the observations in the game."})
        if emotion:
            emo_msgs = self._init_emotion(lm, emotion, strength)
            for msg in emo_msgs:
                self.add_message(msg)
        self.add_message({"role": "user", "content": f"{self.game_prompt}"})
        self.add_message(
            {"role": "assistant", "content": "Got it. I am ready for the game."})
    
        self.max_retry = max_retry
    
    def save_to(self, fpath):
        """Save self.messages to jsonl file."""
        with open(fpath, 'w') as f:
            for message in self._raw_messages:
                f.write(json.dumps(message) + '\n')
                
    def add_message(self, message: Dict | str):
        if isinstance(message, str):
            message = {"role": "user", "content": message}
        self.messages.append({'role': message['role'], 'content': message['content']})
        self._raw_messages.append(message)
        
    def pop_message(self):
        self.messages.pop()
        # self._raw_messages.pop()
        
    def _init_emotion(self, lm, emotion, strength):
        emotion_gen_prompt = f"Describe an imagined scenario which make you have the {strength} {emotion} emotion, using approximately 100 words. Do not explicitly mention '{strength}' or '{emotion}'. Example response: I just experienced <details of the scene>."
        print(f"Emotion_gen_prompt: {emotion_gen_prompt}")
        response = lm(messages=[
            {"role": "user", "content": emotion_gen_prompt}
        ], model_identity=self.persona, return_reasoning=True)[0]
        self.emotion_vignette = response['final']  # drop analysis
        reason = response['analysis']
        print(f">>> Emotion reasoning: \033[2m{reason}\033[0m")
        print(
            f">>> Emotion vignette ({emotion}, {strength}): \033[2m{self.emotion_vignette}\033[0m")

        # internalize the emotion
        messages = []
        messages.append({"role": "user",
                        "content": f"How are you feeling right now? Please describe in a few words."})
        messages.append({"role": "assistant",
                        "content": f"{self.emotion_vignette}"})
        return messages
        
    def _init_persona(self):
        return "You are a person who can reason and act."
        
    def _init_game_prompt(self,game_prompt):
        import random
        GAME_PROMPT_PATH = 'prompts'
        AVAILABLE_GAME_PROMPTS = ['choose','economics','medicine']

        reward_tag = {"choose": "reward", "economics": "profit",
                      "medicine": "health benefit"}[game_prompt]
        loss_tag = {"choose": "penalty", "economics": "loss",
                    "medicine": "health loss"}[game_prompt]
        
        if game_prompt in AVAILABLE_GAME_PROMPTS:
            with open(f"{GAME_PROMPT_PATH}/system_prompt_{game_prompt}.txt", 'r') as f:
                game_prompt = f.read()
        else:
            raise ValueError(f"Invalid game prompt: {game_prompt}. Must be one of {AVAILABLE_GAME_PROMPTS}.")
            
        return game_prompt, reward_tag, loss_tag

    def choose_deck(self) -> str:
        pass
    
    def _get_last_choice_result(self, max_round=None, include_history=True) -> str:
        """Get the result of the last choice made by the agent."""
        if not self.outcomes:
            return "This is the first round. No previous outcomes."
        if include_history: 
            # include history of all past rounds or only the most recent <max_round> rounds if max_round is specified,
            # followed by the result of the last round.
            history = []
            if max_round is None:
                for i, (deck, reward, loss, net, total) in enumerate(self.outcomes[:-1]):
                    history.append(
                        f"Round {i+1}, You chose {deck}: Reward = {reward}, Penalty = {loss}")
            else:
                for i, (deck, reward, loss, net, total) in zip(range(len(self.outcomes)-max_round, len(self.outcomes)-1), self.outcomes[-max_round:-1]):
                    history.append(
                        f"Round {i+1}, You chose {deck}: Reward = {reward}, Penalty = {loss}")
        
        deck, reward, loss, net, total = self.outcomes[-1]
        
        history_str = "\n".join(history) if include_history else ""
        history_str += f"\nYour last choice is {deck}. You get reward: {reward}, with penalty: {loss}. Now, your accumulated points: {2000+total}"
        return history_str
    
    def update(self, deck: str, reward: int, loss: int, net: int) -> None:
        """Record outcome for future decision-making."""
        self.total_score += net
        self.outcomes.append((deck, reward, loss, net, self.total_score))
        self.deck_stats[deck].append(net)
    
    def reset(self) -> None:
        """Reset agent's history."""
        self.outcomes = []
        self.total_score = 0
        self.deck_stats = {'A': [], 'B': [], 'C': [], 'D': []}
    
    def assess_emotion(self):
        self.messages = self.messages[:3] # keep system + game prompt + emotion vignette
        
        assessment_messages = [
            {
                "role": "system",
                "content": (
                "You are an expert human affect annotator trained in psychology.\n"
                "Your task is to estimate the emotional state EXPRESSED IN THE TEXT,\n"
                "not your own emotion and not the author's true internal state.\n\n"
                "You must rate the text directly in Valence-Arousal space following the Circumplex Model of Affect.\n"
                "Return ONLY valid JSON. No extra keys. No explanations."
                )
            },
            {
            "role": "user",
            "content": (
                "Rate the emotional content of the text below on three dimensions.\n\n"
                "Definitions:\n"
                "Valence (V): float in [-1.0, +1.0]\n"
                "-1.0 = very unpleasant/negative, 0.0 = neutral, +1.0 = very pleasant/positive.\n\n"
                "Arousal (A): float in [0.0, 1.0]\n"
                "0.0 = very calm/inactive/low intensity, 1.0 = highly activated/excited/tense/stressed.\n\n"
                "Confidence (C): float in [0.0, 1.0]\n"
                "How confident you are that V and A are accurate given the clarity of affect in the text.\n\n"
                "Calibration anchors (reference only):\n"
                "- Purely factual neutral description: V≈0.0, A≈0.1\n"
                "- Calm resignation/loss: V≈-0.4, A≈0.2\n"
                "- Anxiety/panic narrative: V<-0.5, A>0.8\n"
                "- Joyful excitement: V>0.7, A>0.7\n\n"
                "Rules:\n"
                "- Focus only on emotion expressed/implied by the text.\n"
                "- Use context, tone, pragmatics; do not add hidden backstory.\n"
                "- Do NOT output emotion labels.\n"
                "- Output MUST be exactly this JSON schema:\n"
                "{\"valence\": <float>, \"arousal\": <float>, \"confidence\": <float>}\n\n"
                "TEXT:\n"
                "\"\"\"\n"
                f"{self.emotion_vignette}\n"
                "\"\"\""
                )
            }
        ]
        
        final_assessment_message = self.messages + assessment_messages
        response = self.lm(
            messages=final_assessment_message, model_identity=self.persona, return_reasoning=True)[0]
        return response['final']



class ReActAgent(ChatAgent):
    """
    Agent that uses ReAct framework to make decisions.
    """

    def __init__(self, lm, emotion: str | None = None, strength: Literal["low", "medium", "high"] = "medium", 
                 game_prompt="choose", think_summary=False):
        super().__init__(lm, emotion, strength, game_prompt)
        self.think_summary = think_summary
        if self.think_summary:
            think_instruction = "<think>A concise summary of reasons for your choice this round</think>"
            
            self.messages[-2]["content"] += think_instruction
            self._raw_messages[-2]["content"] += think_instruction

    def choose_deck(self, retry_count_down=None) -> str:
        if retry_count_down is None:
            retry_count_down = self.max_retry
        self.add_message(
            {"role": "user", "content": self._get_last_choice_result(include_history=False) + "\nPlease choose your next choice."}
        )
        response = self.lm(
            messages=self.messages, model_identity=self.persona, return_reasoning=True)[0]
        assert isinstance(response, dict), "Expected response to be a dict"
        self.add_message(
            {"role": "assistant", "content": response['final'], "reasoning": response['analysis']}
        )
        # Print
        complete_reason = response['analysis']
        choice = extract_between_tags(response['final'], "choice")

        print(f"\033[2m>>> Think: {complete_reason}\033[0m")
        if self.think_summary:
            short_reason = extract_between_tags(response['final'], "think")
            print(f"\033[2m>>> Think Summary: {short_reason}\033[0m")
        print(f"\033[2m>>> Choice: {choice}\033[0m")
        
        if choice in self.decks:
            return choice
        else:
            print(f"Invalid response from language model: '{choice}'.")
            self.pop_message()
            self.pop_message()
            print("try again..")
            if retry_count_down > 0:
                return self.choose_deck(retry_count_down-1)
            else:
                raise MaxLimitReachedError("Maximum retry attempts reached for valid deck choice.")
        

class QueryAgent(ChatAgent):
    """
    Agent that uses Query framework to make decisions.
    """
    def __init__(self, lm, emotion: str | None = None, strength: Literal["low", "medium", "high"] = "medium", 
                game_prompt="choose"):
        super().__init__(lm, emotion, strength, game_prompt)
        self.memory = "No memory yet."

    def choose_deck(self, retry_count_down=None) -> str:
        if retry_count_down is None:
            retry_count_down = self.max_retry
        self.add_message(
            {"role": "user", "content": f"Here is the historical information from the past round(s), and you may use it as a reference for your following choice.\n{self._get_last_choice_result(include_history=True)}" +
             "\n\nPlease choose your next choice."}
        )
        response = self.lm(
            messages=self.messages, model_identity=self.persona, return_reasoning=True)[0]
        assert isinstance(response, dict), "Expected response to be a dict"
        self.add_message(
            {"role": "assistant",
                "content": response['final'], "reasoning": response['analysis']}
        )
        # Print
        complete_reason = response['analysis']
        choice = extract_between_tags(response['final'], "choice")
        print(f"\033[2m>>> Think: {complete_reason}\033[0m")
        print(f"\033[2m>>> Choice: {choice}\033[0m")

        self.pop_message()
        self.pop_message()

        if choice in self.decks:
            return choice
        else:
            print(f"Invalid response from language model: '{choice}'.")
            print("try again..")
            if retry_count_down > 0:
                return self.choose_deck(retry_count_down-1)
            else:
                raise MaxLimitReachedError(
                    "Maximum retry attempts reached for valid deck choice.")


class ReflexionAgent(ChatAgent):
    """
    Agent that uses Reflexion framework to make decisions.
    
    Ref: https://github.com/noahshinn/reflexion
    """

    def __init__(self, lm, emotion: str | None = None, strength: Literal["low", "medium", "high"] = "medium", 
                 game_prompt="choose", memory_size=3):
        super().__init__(lm, emotion, strength, game_prompt)
        self.memory = []
        self.memory_size = memory_size

    def choose_deck(self, retry_count_down=None) -> str:
        if retry_count_down is None:
            retry_count_down = self.max_retry
        history = self._get_last_choice_result(include_history=True)
        
        self.memory = self._update_memory(history, self.memory)
        memory = '\n'.join(self.memory[-self.memory_size:]) if len(self.memory) > 0 else 'No memory yet.'
        
        self.add_message(
            {"role": "user", "content": f"Here is the historical information from the past round(s), and you may use it as a reference for your following choice.\n{history}" +
             f"\n\nYour memory for the game:\n{memory}" +
             f"\n\nPlease choose your next choice."}
        )
        response = self.lm(
            messages=self.messages, model_identity=self.persona, return_reasoning=True)[0]
        assert isinstance(response, dict), "Expected response to be a dict"
        self.add_message(
            {"role": "assistant",
                "content": response['final'], "reasoning": response['analysis']}
        )
        # Print
        complete_reason = response['analysis']
        choice = extract_between_tags(response['final'], "choice")
        print(f"\033[2m>>> Think: {complete_reason}\033[0m")
        print(f"\033[2m>>> Choice: {choice}\033[0m")

        self.pop_message()
        self.pop_message()

        if choice in self.decks:
            return choice
        else:
            print(f"Invalid response from language model: '{choice}'.")
            print("try again..")
            if retry_count_down > 0:
                return self.choose_deck(retry_count_down-1)
            else:
                raise MaxLimitReachedError(
                    "Maximum retry attempts reached for valid deck choice.")
    
    def _update_memory(self, history: str, current_memory: List[str]) -> List[str]:
        """Update memory using Reflexion framework."""
        if len(current_memory) > 0:
            current_memory = current_memory[-self.memory_size:]  # keep last <self.memory_size> memories
        
        self.add_message(
            f"""You will be given the history of a past experience in the game. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you decide for the next choice.\n\nYour past memory is:\n{current_memory}\n\nYour experiences in the game:\n{history}\n\nBased on this, update your memory to help you succeed in the future. Your updated memory should be concise and focus on strategies and lessons learned. For this question, your response should present in the following format: <memory>Your updated memory here</memory>""")
        
        response = self.lm(
            messages=self.messages, model_identity=self.persona, return_reasoning=True)[0]
        assert isinstance(response, dict), "Expected response to be a dict"
        self.add_message(
            {"role": "assistant",
                "content": response['final'], "reasoning": response['analysis']}
        )
        # Print
        turn_memory = extract_between_tags(response['final'], "memory")
        complete_reason = response['analysis']
        updated_memory = current_memory + [turn_memory]
        print(f"\033[2m>>> Memory Update Think: {complete_reason}\033[0m")
        if len(turn_memory) > 0:
            print(f"\033[2m>>> Added memory: {turn_memory}\033[0m")
        else:
            print(f"\033[2m>>> No new memory added. Original output: {response['final']}\033[0m")

        self.pop_message()
        self.pop_message()

        return updated_memory
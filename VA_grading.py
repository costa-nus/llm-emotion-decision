import os
os.makedirs('out', exist_ok=True)

from transformers import set_seed
from utils.hf import load_chat_lm
import argparse
import json
from utils.agents import ChatAgent

# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model' , type=str, default='openai/gpt-oss-20b',
                    help='The LLM model to use for ranking.')
parser.add_argument('--randomized' , action='store_true',
                    help='Whether to use randomized model settings.')
parser.add_argument('--seed' , type=int, default=1,
                    help='Random seed for reproducibility.')
parser.add_argument('--emotion', type=str, default='anxiety',
                    choices=['anxiety', 'sadness', 'anger', 'happiness'],
                    help='Emotion for agent (default: anxiety)')
parser.add_argument('--reasoning_effort', type=str, default='Low',
                    choices=['No', 'Low', 'Medium', 'High'],
                    help='The reasoning effort level for the model.')
args = parser.parse_args()

model_hf_name = args.model
randomized = args.randomized
emotion = args.emotion
seed = args.seed          
set_seed(args.seed)

if 'gpt' not in model_hf_name and randomized:
    if 'llama' in model_hf_name: 
        temperature = 0.6
        top_p = 0.9
    else: 
        temperature = 0.7
        top_p = 0.8
    gen_args = {'do_sample': True,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': 20,
            'min_p': 0,}
elif model_hf_name == 'openai/gpt-oss-20b' and randomized:
    gen_args = {'do_sample': True,
            'temperature':0.9,
            'reasoning_effort': args.reasoning_effort,}
else:
    gen_args = {'do_sample': False,
            'temperature': 0.0,}
    
lm = load_chat_lm(model_hf_name,
                    max_new_tokens=32768,
                    **gen_args
                )
agent = ChatAgent(lm, emotion=emotion)

with open('out/VA_grading.jsonl', 'a') as f:
    f.write(json.dumps({
        'model': model_hf_name,
        'emotion': args.emotion,
        'seed': args.seed,
        'emotion_vignette': agent.emotion_vignette,
        'assessment': json.loads(agent.assess_emotion()),
    })+'\n')



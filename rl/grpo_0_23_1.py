# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os, sys
sys.path.append(os.getcwd())
from os.path import join
import re
import random
random.seed(42)
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional
from huggingface_hub import login
from typing import List, Optional
from dataclasses import asdict
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor

from math_verify import parse, verify
from rl.trainer.grpo_trainer_0_23_1 import GRPOTrainer
from rl.trainer.grpo_config_0_23_1 import GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

char_list = [chr(ord('A') + i) for i in range(26)]


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy",],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default = 64 * 28 * 28,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default = 64 * 28 * 28,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    max_frames: Optional[int] = field(default = 30)
    '''arpo args'''
    attn_enhanced: Optional[bool] = field(default=False)
    max_weight: Optional[float] = field(default=1.7)
    select_strategy: Optional[str] = field(default='topk-union')
    select_level: Optional[str] = field(default='hard')
    token_nums_for_avg_frame_score: Optional[int] = field(default=15)
    frame_nums_for_select_valid_frames: Optional[int] = field(default=5)
    token_nums_for_update_per_frame: Optional[int] = field(default=30)
    add_one_to_weights: Optional[bool] = field(default = True)



'''image, math problem reward'''
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards


'''video, nextqa qa reward'''
def nextqa_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>\s*\([A-Z]\)\s*</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


def nextqa_accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        matches = re.findall(r'<answer>\s*(\([A-Z]\))\s*</answer>', completion)
        if len(matches) == 1: 
            pred = matches[0].strip()
        else: 
            rewards.append(0)
            continue

        matches = re.findall(r'\(.*?\)', sol)
        if len(matches) == 1: 
            gt = matches[0].strip()
        else:
            rewards.append(0)
            continue
            
        if pred == gt:
            rewards.append(1)
        else: 
            rewards.append(0)

    return rewards


def video_r1_accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        matches = re.findall(r'<answer>\s*([A-Z])\s*</answer>', completion)
        if len(matches) == 1:
            pred = matches[0].strip()
        else:
            rewards.append(0)
            continue

        matches = re.findall(r'<answer>([A-Z])</answer>', sol) # A
        if len(matches) == 1: 
            gt = matches[0].strip()
        else:
            rewards.append(0)
            continue
            
        if pred == gt:
            rewards.append(1)
        else: 
            rewards.append(0)

    return rewards


def video_r1_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>\s*[A-Z]\s*</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


'''time iou rewards'''
def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<time>(.*?)</time>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    # print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) to (\d+\.?\d*)", last_answer_content, re.IGNORECASE
    )

    if not matches:
        return None

    pred_time = []
    for mat in matches:
        start_time = float(mat[0])
        end_time = float(mat[1])
        pred_time.append([start_time, end_time])
    # return start_time, end_time
    return pred_time


def nextgqa_tiou_reward(
    completions, solution, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    time_clues = kwargs['time_clues']
    duration = kwargs['duration']
    for content, sol, time_clue, dur in zip(completions, solution, time_clues, duration):  # Added video_durations
        reward = 0.0
        try:
            pred_times = parse_timestamp_output(content)
            if pred_times is not None:
                gt_flags = [0] * (int(dur) + 10)
                for clue in time_clue:
                    for t in range(int(clue[0]), int(clue[1])): gt_flags[t] = 1
                
                pred_flags = [0] * (int(dur) + 10)
                for pred_time in pred_times:
                    for t in range(int(pred_time[0]), int(pred_time[1])): 
                        if t < len(pred_flags): pred_flags[t] = 1
                
                inception, union = 0, 0
                for t in range(len(gt_flags)):
                    if gt_flags[t] == 1 or pred_flags[t] == 1: union += 1
                    if gt_flags[t] == 1 and pred_flags[t] == 1: inception += 1
                
                iou = inception / (union + 1e-8)

                reward += iou
        except Exception as e:
            print(f'==> nextgqa tiou error. info: {str(e)}, skip...')
            continue

        rewards.append(reward)

    return rewards


def nextgqa_accuracy_reward(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        matches = re.findall(r'<answer>\s*([A-Z])\s*</answer>', completion)
        if len(matches) == 1:
            pred = matches[0].strip()
        else:
            rewards.append(0)
            continue

        matches = re.findall(r'<answer>([A-Z])</answer>', sol) # A
        if len(matches) == 1: 
            gt = matches[0].strip()
        else:
            rewards.append(0)
            continue
            
        if pred == gt:
            rewards.append(1)
        else: 
            rewards.append(0)

    return rewards


def nextgqa_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*<time>.*?</time>$", re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


'''think length rewards'''
def extract_think_content(completion: str) -> Optional[str]:
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


def reward_think_length(
    completions: List[str],
    solution,
    weight: float = 0.001,
    max_length: int = 256,
    **kwargs,
) -> List[float]:
    rewards = []
    for completion, sol in zip(completions, solution):
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            think_length = len(think_content.split(' '))
            capped_length = min(think_length, max_length)
            score = weight * capped_length
        else:
            score = 0.0

        ### answer correctness
        matches = re.findall(r'<answer>\s*([A-Z])\s*</answer>', completion)
        if len(matches) == 1:
            pred = matches[0].strip()
        else:
            pred = None
            score = 0.0

        matches = re.findall(r'<answer>([A-Z])</answer>', sol) # A
        if len(matches) == 1: 
            gt = matches[0].strip()
        else:
            gt = '-1'
            score = 0.0
            
        flag = 1 if pred == gt else 0
        score = score * flag

        rewards.append(max(0.0, score))
        
    return rewards



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    'nextqa_accuracy': nextqa_accuracy_reward,
    'nextqa_format': nextqa_format_reward,
    'video_r1_accuracy': video_r1_accuracy_reward,
    'video_r1_format': video_r1_format_reward,
    'nextgqa_tiou': nextgqa_tiou_reward,
    'nextgqa_accuracy': nextgqa_accuracy_reward,
    'nextgqa_format': nextgqa_format_reward,
    'think_length_reward': reward_think_length
}


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

nextqa_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer>(A)</answer>"
)


QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "


NEXTGQA_QUESTION_TEMPLATE = '''
The video is {duration} seconds long, with {frame_nums} frames evenly sampled from it. 
Based on the video content, think about the question deeply, select the correct option, and provide ONE or MORE time periods in the video where the clues corresponding to the correct option occur.
Question: {Question} 
Output the thinking process in <think> </think> tags, final answer in <answer> </answer> tags and time clues in <time> </time> tags, i.e., <think> reasoning process here </think> <answer> answer here </answer> <time> 8.4 to 12.5, 25 to 30.2 </time>
'''

from datasets import Dataset, DatasetDict
import json


def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def make_conversation_egoplan(example, data_root_dir, processor: Qwen2_5_VLProcessor):
    if 'golden_choice_idx' not in example:
        negative_answers = random.sample(example["negative_answers"], 3)
        options = negative_answers + [example["answer"]]
    else:
        options = [example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']]

    random.shuffle(options)
    answer_index = options.index(example["answer"])
    problem = f"{example['question']}\n" + "\n".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)]) + "\n"
    solution = f"<answer> ({chr(65 + answer_index)}) </answer>"
    
    content = []
    if len(example['task_progress_metadata']) > 0:
        video_path = os.path.join(data_root_dir, 'videos', example['video_source'], example['video_basename'])
        content.append({"type": "video", "video": video_path})
    else:
        video_path = None
    
    image_path = os.path.join(data_root_dir, 'images', example['video_source'], example['current_observation_basename'])
    content.extend([
        {"type": "image", "image": image_path},
        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
    ])

    conversation = [
        {
            "role": "user",
            "content": content
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    return {
        "prompt": prompt,
        "video": video_path,
        'image': image_path,
        'solution': solution
    }



local_rank=None

def main(script_args, training_args, model_args):
    global local_rank
    local_rank = training_args.local_rank
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    output_dir = training_args.output_dir
    save_config = {
        'script_args':asdict(script_args),
        'training_args':asdict(training_args),
        'model_args': asdict(model_args)
    }
    os.makedirs(output_dir,exist_ok=True)
    with open(join(output_dir,'saved_config.json'),'w') as f:
        f.write(json.dumps(save_config,indent=4))
    
    
    ### training dataset
    dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    train_dataset = dataset['train']

    model_id = model_args.model_name_or_path
    min_pixels = script_args.min_pixels
    max_pixels = script_args.max_pixels
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast = True, padding_side="left",
                                                    min_pixels = min_pixels, max_pixels = max_pixels)



    def make_conversation_video_r1(example):
        problem = example['problem']
        options = example['options']
        solution = example['solution']
        path = example['path']
        video_path = os.path.join('', path[2:])

        problem = problem + '\n' + '\n'.join(options)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question = problem)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "video": video_path,
            'solution': solution,
        }



    def make_conversation_perception_test(example):
        problem = example['problem']
        options = example['options']
        solution = example['solution']
        video_path = example['path']
        
        problem = problem + '\n' + '\n'.join(options)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question = problem)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "video": video_path,
            'solution': solution
        }


    def make_conversation_nextqa(example, processor: Qwen2_5_VLProcessor):
        video_path = example['video_path']
        problem = example['problem']
        solution = example['solution']
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question = problem)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return {
            "prompt": prompt,
            "video": video_path,
            'solution': solution
        }


    def make_conversation_seed_bench_r1(example):
        data_root_dir = ''
        if 'golden_choice_idx' not in example:
            negative_answers = random.sample(example["negative_answers"], 3)
            options = negative_answers + [example["answer"]]
        else:
            options = [example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']]

        random.shuffle(options)
        answer_index = options.index(example["answer"])
        problem = f"{example['question']}\n" + "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"
        solution = f"<answer>{chr(65 + answer_index)}</answer>"
        
        content = []
        if len(example['task_progress_metadata']) > 0:
            video_path = os.path.join(data_root_dir, 'videos', example['video_source'], example['video_basename'])
            content.append({"type": "video", "video": video_path})
        else:
            video_path = None
        
        image_path = os.path.join(data_root_dir, 'images', example['video_source'], example['current_observation_basename'])
        content.extend([
            {"type": "image", "image": image_path},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
        ])

        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return {
            "prompt": prompt,
            "video": video_path,
            'image': image_path,
            'solution': solution
        }
    

    def make_conversation_nextgqa(example, max_frames):
        problem = example['problem']
        solution = example['solution']
        video_path = example['video_path']
        duration = example['duration']
        frame_nums = min(max_frames, int(duration) // 2) # fps = 1.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": NEXTGQA_QUESTION_TEMPLATE.format(duration = duration, frame_nums = frame_nums, Question = problem)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return {
            "prompt": prompt,
            "video": video_path,
            'solution': solution
        }


    def make_conversation_dynamic_seed_bench_r1(example):
        problem = example['problem']
        solution = example['solution']
        video_path = example['video_path']
        image_path = example['image_path']

        content = []
        content.append({"type": "video", "video": video_path})
        content.extend([
            {"type": "image", "image": image_path},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question = problem)},
        ])
        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "video": video_path,
            'image': image_path,
            'solution': solution
        }


    def make_conversation(example):
        dataset_name = example['dataset_name']
        if dataset_name == 'nextqa':
            return make_conversation_nextqa(example, processor)

        elif dataset_name == 'seed_bench_r1':
            return make_conversation_seed_bench_r1(example)
        
        elif dataset_name == 'video_r1':
            return make_conversation_video_r1(example)

        elif dataset_name == 'Perception-test':
            return make_conversation_perception_test(example)

        elif dataset_name == 'next-gqa':
            return make_conversation_nextgqa(example, script_args.max_frames)
        

        elif dataset_name == 'dynamic_sampling_seed_bench_r1':
            return make_conversation_dynamic_seed_bench_r1(example)
        

        else:
            raise ValueError(f'invalid dataset name: [{dataset_name}]')


    train_dataset = train_dataset.map(make_conversation)
    
    trainer_cls = GRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model = model_id,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = None,
        processing_class = processor,
        peft_config = None,
        min_pixels = min_pixels,
        max_pixels = max_pixels,
        attn_enhanced = script_args.attn_enhanced,
        attn_layers = [-1, -2, -3],
        max_weight = script_args.max_weight,
        select_strategy = script_args.select_strategy,
        select_level = script_args.select_level,
        token_nums_for_avg_frame_score = script_args.token_nums_for_avg_frame_score,
        frame_nums_for_select_valid_frames = script_args.frame_nums_for_select_valid_frames,
        token_nums_for_update_per_frame = script_args.token_nums_for_update_per_frame,
        add_one_to_weights = script_args.add_one_to_weights,
        max_frames = script_args.max_frames
    )

    # Train and push the model to the Hub
    trainer.train()

    
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

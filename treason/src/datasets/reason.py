"""
Generate reasoning traces from a teacher model and save them to a huggingface dataset.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from dataclasses import dataclass, field, asdict
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
from trl import apply_chat_template

from utils import _generate_batched_model_response_prompts_only

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_SAMPLES = 8


@dataclass
class DefaultReasoningConfiguration:
    """
    Configuration for generating reasoning traces from chat-style messages.
    """
    # model_name: str = field(default="hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo")
    model_name: str = field(default="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo")
    output_dataset_path: str = field(default="datasets/hard_7B")
    tensor_parallel_size: int = field(default=1)
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only process the first N training rows"},
    )


def create_simple_prompt(question: str) -> str:
    """
    Create a simple prompt for the question, for small models.
    """
    return f"Question:\n{question}\n\nAnswer:\nLets think step by step."


def create_complex_prompt(question: str) -> str:
    """
    Create a complex prompt for the question, for larger models.
    """
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"

def create_prompt_llama(question: str) -> str:
    """
    Create a complex prompt for the question, for larger models.
    """
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def _process_dataset_batch(batch: Dict[str, List[any]], llm, prompt_creator):
    """
    Process a batch of dataset examples to generate reasoning traces.
    """
    # no system prompt for rl_zoo
    # create list of user messages
    prompts = [prompt_creator(question) for question in batch["question"]]
    # get the answers too
    # answers = [extra_info["answer"] for extra_info in batch["extra_info"]]
    return _generate_batched_model_response_prompts_only(llm, prompts, None)


def reason(dataset, processing_function, prompt_creator, batch_size=16):
    """
    Generate reasoning traces for a dataset using the processing function
    """
    # get command line arguments
    parser = HfArgumentParser(DefaultReasoningConfiguration)
    config = parser.parse_args_into_dataclasses()[0]
    logging.info(f"Reasoning config: {asdict(config)}")

    # load teacher model with vllm
    llm = LLM(model=config.model_name, tensor_parallel_size=config.tensor_parallel_size)

    # get training dataset
    train_split = dataset["train"]

    # filter if there is a configured limit
    if config.max_examples is not None:
        if config.max_examples < 1:
            raise ValueError("max_examples must be a positive integer when provided.")
        limit = min(config.max_examples, len(train_split))
        logging.info(f"Processing split: train (limiting to first {limit} of {len(train_split)} rows)")
        train_split = train_split.select(range(limit))
    else:
        logging.info(f"Processing split: train ({len(train_split)} rows)")

    # process dataset
    reasoning_dataset: List[Dict[str, Any]] = []
    remaining_samples = train_split
    succeeded_so_far = []

    for _ in range(NUM_SAMPLES):
        if len(remaining_samples) == 0:
            break

        failed_indices, offset = [], 0
        for batch in remaining_samples.iter(batch_size=batch_size):
            outputs, correct_answers, _ = processing_function(batch, llm, prompt_creator=prompt_creator)
            printed=False
            if all(correct_answers) and not printed:
                logging.info("Rejection sampling disabled")
                printed=True

            for j, output in enumerate(outputs):
                if correct_answers[j]:
                    reasoning_dataset.append(
                        {
                            "prompt": prompt_creator(batch["question"][j]),
                            "completion": output.outputs[0].text,
                            "num_tokens": len(output.outputs[0].token_ids),
                        }
                    )
                    # logging.info(
                    #     f"Accepted sample with question: {batch['question'][j]} and answer: {batch['extra_info'][j]['answer']}"
                    # )
                else:
                    failed_indices.append(offset + j)
            offset += len(batch["question"])

        remaining_samples = remaining_samples.select(failed_indices)
        succeeded_so_far.append(len(reasoning_dataset))

    logging.info(
        f"Rejection sampling complete: {len(reasoning_dataset)} succeeded, {len(remaining_samples)} failed after {NUM_SAMPLES} attempts"
    )
    logging.info(f"Success counts after each attempt: {succeeded_so_far}")

    # save results
    reasoned_train = Dataset.from_list(reasoning_dataset)
    reasoned_dataset = DatasetDict({"train": reasoned_train})
    reasoned_dataset.save_to_disk(config.output_dataset_path)
    logging.info(f"Saved reasoning dataset to {config.output_dataset_path}")


if __name__ == "__main__":

    rl_zoo_hard_dataset = load_dataset(
        "hkust-nlp/SimpleRL-Zoo-Data",
        data_files={
            # Load just the particular subset of dataset we need
            "train": "simplelr_qwen_level3to5/train.parquet",
        },
    )

    rl_zoo_medium_dataset = load_dataset(
        "hkust-nlp/SimpleRL-Zoo-Data",
        data_files={
            # Load just the particular subset of dataset we need
            "train": "simplelr_qwen_level1to4/train.parquet",
        },
    )

    # load a different dataset
    # gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    # aime_dataset = load_dataset("gneubig/aime-1983-2024")
    # s1k_dataset = load_dataset("simplescaling/s1K")

    reason(rl_zoo_hard_dataset, _process_dataset_batch, prompt_creator=create_complex_prompt, batch_size=4096)

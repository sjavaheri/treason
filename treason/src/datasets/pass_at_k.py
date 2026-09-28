"""
Pass@k evaluation for a model on the MATH-500 dataset

Designed and written by Shidan Javaheri, partial code completion by Claude Opus 4.6 - 5.5
"""

import json
import logging
import os
import numpy as np

from typing import Dict, List, Optional
from gc import collect
from dataclasses import dataclass, field, asdict
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from utils import _generate_batched_model_response_prompts_only
from minerva_math_utils import last_boxed_only_string, remove_boxed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _estimator_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Numerically stable estimator for pass@k.
    https://github.com/huggingface/evaluate/blob/main/metrics/code_eval/code_eval.py#L198
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


NUM_SAMPLES = 256
BATCH_SIZE = 16

MINERVA_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class DefaultReasoningConfiguration:
    """
    Configuration for generating reasoning traces from chat-style messages.
    """
    tensor_parallel_size: int = field(default=1)
    no_of_qs: Optional[int] = field(
        default=None,
        metadata={"help": "If set, selects a random subset of N questions to process, for each difficulty level"},
    )
    model_name: str = field(
        default="/scratch/a6y/sjavaheri.a6y/models/Qwen2.5-3B_hard_3B",
        metadata={"help": "Model name or path to evaluate"}
    )
    display_name: str = field(
        default="lam_1_e1",
        metadata={"help": "Display name for saving results"}
    )
    dataset_name: str = field(
        default="math_500",
        metadata={"help": "Dataset to use (math_500, gsm8k or minerva_math)"}
    )
    use_complex_prompt: bool = field(
        default=False,
        metadata={"help": "Set to true to use the complex prompt"}
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


def _process_dataset_batch(batch: Dict[str, List[any]], llm, prompt_creator, question_key, answer_key):
    """
    Process a batch of dataset examples to generate reasoning traces.
    """
    # no system prompt for rl_zoo
    # create list of user messages
    prompts = [prompt_creator(question) for question in batch[question_key]]
    answers = [answer for answer in batch[answer_key]]
    return _generate_batched_model_response_prompts_only(llm, prompts, answers, samples=NUM_SAMPLES)


def load_minerva_math() -> DatasetDict:
    """
    Load the full MATH test set (5000 questions) used by minerva_math, across all subjects.
    """
    test_split = concatenate_datasets(
        [load_dataset("EleutherAI/hendrycks_math", subject, split="test") for subject in MINERVA_MATH_SUBJECTS]
    )
    # hendrycks_math has no answer column, so extract it from the boxed solution
    test_split = test_split.map(lambda doc: {"answer": remove_boxed(last_boxed_only_string(doc["solution"]))})
    return DatasetDict({"test": test_split})


def reason(dataset, processing_function, prompt_creator, question_key="question", answer_key="answer"):
    """
    Generate reasoning traces for a dataset using the processing function
    """
    # get command line arguments
    parser = HfArgumentParser(DefaultReasoningConfiguration)
    config = parser.parse_args_into_dataclasses()[0]
    logging.info(f"Reasoning config: {asdict(config)}")

    # get training dataset
    test_split = dataset["test"]

    # filter if there is a configured limit
    if config.no_of_qs is not None:
        if config.no_of_qs < 1:
            raise ValueError("no_of_qs must be a positive integer when provided.")
        limit = min(config.no_of_qs, len(test_split))
        logging.info(f"Processing split: test (limiting to first {limit} of {len(test_split)} rows)")
        test_split = test_split.select(range(limit))
    else:
        logging.info(f"Processing split: test ({len(test_split)} rows)")

    # create figures directory at project root, with a subdirectory per model
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "pass_at_k")
    os.makedirs(figures_dir, exist_ok=True)

    # k values to evaluate (1 through NUM_SAMPLES //2 )
    k_values = np.arange(1, NUM_SAMPLES // 2 + 1)

    model_name = config.model_name
    display_name = config.display_name


    logging.info(f"Loading model: {display_name} ({model_name})")
    llm = LLM(model=model_name, tensor_parallel_size=config.tensor_parallel_size, gpu_memory_utilization=0.85)

    # added by opus 5.5
    # collect per-question correct counts (c values)
    # resume from a checkpoint of per-question c values, if one exists
    ckpt_path = os.path.join(figures_dir, f"ckpt_{display_name}_{config.dataset_name}.json")
    c_values, c_values_flexible = [], []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            c_values, c_values_flexible = json.load(f)
        logging.info(f"Resuming from {ckpt_path}: {len(c_values)} questions already done")
    remaining = test_split.select(range(len(c_values), len(test_split)))

    for _, batch in enumerate(remaining.iter(batch_size=BATCH_SIZE)):
        # get number of questions. Might be less than batch size in last batch
        n_qs = len(batch[question_key])
        _, correct_answers, correct_answers_flexible = processing_function(batch, llm, prompt_creator=prompt_creator, question_key=question_key, answer_key=answer_key)

        # correct_answers is flat: [q0_s0, q0_s1, ..., q0_s(N-1), q1_s0, ...]
        correct_arr = np.array(correct_answers, dtype=int).reshape(n_qs, NUM_SAMPLES)
        correct_arr_flexible = np.array(correct_answers_flexible, dtype=int).reshape(n_qs, NUM_SAMPLES)
        # sum across samples to get c values for each question
        c_values.extend(correct_arr.sum(axis=1).tolist())
        c_values_flexible.extend(correct_arr_flexible.sum(axis=1).tolist())
        with open(ckpt_path, "w") as f:
            json.dump([c_values, c_values_flexible], f)

    logging.info(f"{display_name}: {len(c_values)} questions, mean c = {np.mean(c_values):.2f}")


    # compute pass@k for each k, averaged across all questions
    pass_at_k_values = np.zeros(len(k_values))
    pass_at_k_values_flexible = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        per_q = [_estimator_pass_at_k(NUM_SAMPLES, c, k) for c in c_values]
        pass_at_k_values[i] = np.mean(per_q)
        per_q_flexible = [_estimator_pass_at_k(NUM_SAMPLES, c, k) for c in c_values_flexible]
        pass_at_k_values_flexible[i] = np.mean(per_q_flexible)

    # clean up GPU memory
    destroy_model_parallel()
    del llm
    collect()
    logging.info(f"Finished processing model {display_name}.")

    # --- save data ---
    save_path = os.path.join(figures_dir, f"pass_at_k_{display_name}.npz")
    flexible_save_path = os.path.join(figures_dir, f"pass_at_k_flexible_{display_name}.npz")
    np.savez(save_path, k_values=k_values, pass_at_k=pass_at_k_values, display_name=display_name)
    np.savez(flexible_save_path, k_values=k_values, pass_at_k_flexible=pass_at_k_values_flexible, display_name=display_name)
    logging.info(f"Saved pass@k data to {save_path}")
    logging.info(f"Saved flexible pass@k data to {flexible_save_path}")
    os.remove(ckpt_path)


if __name__ == "__main__":
    parser = HfArgumentParser(DefaultReasoningConfiguration)
    config = parser.parse_args_into_dataclasses()[0]

    if config.use_complex_prompt: 
        prompt_creator=create_complex_prompt
        print("Complex prompt selected for dataset " + config.dataset_name)
    else:
        prompt_creator=create_simple_prompt
        print("Simple prompt selected for dataset " + config.dataset_name)

    if config.dataset_name == "math_500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        reason(dataset, _process_dataset_batch, prompt_creator=prompt_creator, question_key="problem", answer_key="answer")
    elif config.dataset_name == "gsm8k":
        dataset = load_dataset("skrishna/gsm8k_only_answer")
        reason(dataset, _process_dataset_batch, prompt_creator=prompt_creator, question_key="text", answer_key="label")
    elif config.dataset_name == "minerva_math":
        dataset = load_minerva_math()
        reason(dataset, _process_dataset_batch, prompt_creator=prompt_creator, question_key="problem", answer_key="answer")
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}. Expected math_500, gsm8k or minerva_math.")

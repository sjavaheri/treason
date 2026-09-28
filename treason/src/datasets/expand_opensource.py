"""
Generate a expanded dataset of reasoning traces from summaries
"""

import logging
import re
from typing import Any, Dict, Optional, List

from dataclasses import dataclass, field, asdict
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from vllm import LLM

from utils import _generate_responses_expansion_rejection_sampling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DefaultExpansionConfiguration:
    """
    Default configurations for distillation on the model
    """
    model_name: str = field(default="meta-llama/Llama-3.2-3B-Instruct")
    dataset_path: str = field(default="treason/datasets/Qwen2.5-14B-SimpleRLZoo-summary")
    output_path: str = field(default="treason/datasets/open-source-expand")
    tensor_parallel_size: int = field(default=1)
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only summarize the first N training rows"},
    )


def _expand_with_same_basemodel(batch: Dict[str, Any], llm: LLM):
    """
    Expand traces by providing them as examples to the same base model
    """
    # get the prompts from the dataset
    prompts_with_notes = []
    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        # split prompt to get question
        _, _, after_user = prompt.partition("<|im_start|>user\n")
        question, _, _ = after_user.partition("\nPlease reason")
        if len(completion) > 2048:
            logging.warning(f"Completion is too long ({len(completion)} tokens), truncating to 2048 tokens.")
            completion = completion[:2048]

        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}
Here is a correct, summarized solution to this problem:
----
{completion}
----

You are an expert at solving math problems. Write out a very natural solution to the problem as if you were solving the problem yourself for the first time, but that relies on the reasoning in the summary above.

- Do not refer to "the notes," "the solution above," or this text in any way — write entirely in your own voice, as an original and natural derivation with no trace of a second source.
- Every intermediate value, equation, and the final answer must match those given above — do not recompute them differently or take a different approach. Your job is to show the full working that justifies the solution.
Ensure that the final answer is formatted in the same way as and exactly matches the final answer in the solution given. Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        prompts_with_notes.append(llama_prompt)

    return _generate_responses_expansion_rejection_sampling(llm, prompts_with_notes)


def expand(processing_function, batch_size=128):
    """
    Expand the reasoning traces in a dataset
    """
    # get command line arguments
    parser = HfArgumentParser(DefaultExpansionConfiguration)
    config = parser.parse_args_into_dataclasses()[0]
    logging.info(f"Expansion Config: {asdict(config)}")

    # load the dataset from the disk
    dataset = load_from_disk(config.dataset_path)
    train_split = dataset["train"]

    # load model and tokenizer with vllm
    llm = LLM(model=config.model_name, tensor_parallel_size=config.tensor_parallel_size)

    # limit the number of examples if specified
    if config.max_examples is not None:
        if config.max_examples < 1:
            raise ValueError("max_examples must be a positive integer when provided.")
        limit = min(config.max_examples, len(train_split))
        logging.info(f"Processing split: train (limiting to first {limit} of {len(train_split)} rows)")
        train_split = train_split.select(range(limit))
    else:
        logging.info(f"Processing split: train ({len(train_split)} rows)")

    NUM_SAMPLES = 128

    # process dataset
    expanded_dataset: List[Dict[str, Any]] = []
    remaining_samples = train_split
    succeeded_so_far = []

    for _ in range(NUM_SAMPLES):
        if len(remaining_samples) == 0:
            break

        failed_indices, offset = [], 0
        for batch in remaining_samples.iter(batch_size=batch_size):
            # get llm outputs
            outputs, correct_answers, _ = processing_function(batch, llm)
            printed=False
            if all(correct_answers) and not printed:
                logging.info("Rejection sampling disabled")
                printed=True
            # process and add to dataset
            for j, output in enumerate(outputs):
                if correct_answers[j]:
                    original_prompt = batch["prompt"][j]
                    original_summary = batch["completion"][j]
                    expansion = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                    expanded_dataset.append(
                        {
                            "prompt": original_prompt,
                            "completion": expansion,
                            "summary_trace": original_summary,
                            "num_tokens": num_tokens,
                        }
                    )
                else:
                    failed_indices.append(offset + j)
            offset += len(batch["prompt"])

        remaining_samples = remaining_samples.select(failed_indices)
        succeeded_so_far.append(len(expanded_dataset))

    logging.info(
        f"Rejection sampling complete: {len(expanded_dataset)} succeeded, {len(remaining_samples)} failed after {NUM_SAMPLES} attempts"
    )
    logging.info(f"Success counts after each attempt: {succeeded_so_far}")

    # save results
    expanded_train = Dataset.from_list(expanded_dataset)
    expanded_dataset = DatasetDict({"train": expanded_train})
    expanded_dataset.save_to_disk(config.output_path)
    logging.info(f"Saved expanded dataset to {config.output_path}")


if __name__ == "__main__":
    expand(batch_size=4096, processing_function=_expand_with_same_basemodel)

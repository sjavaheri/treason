"""
Generate a dataset of synthesized summaries and final answers from close source models
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
    dataset_path: str = field(default="treason/datasets/flash-36-summary")
    output_path: str = field(default="treason/datasets/gemini-stage-1")
    tensor_parallel_size: int = field(default=1)
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only summarize the first N training rows"},
    )


def create_simple_prompt(question: str) -> str:
    """
    Create a simple prompt for the question, for small models.
    """
    return f"Question:\n{question}\n\nAnswer:\nLets think step by step.\n"


def _expand_with_model(batch: Dict[str, Any], llm: LLM):
    """
    Expand traces by providing them as examples to the same base model
    """

    # get the prompts from the dataset
    prompts_with_notes = []
    for question, summary, final_answer in zip(batch["question"], batch["summary"], batch["final_answer"]):
        if len(summary) > 15000:
            logging.warning(f"Completion is too long ({len(summary)} tokens), truncating to roughly 15000 tokens.")
            summary = summary[:15000]

        llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

You are a reasoning synthesizer, whose job is to take a reasoning summary and final answer that solves a problem, and synthesize it into coherent summary of reasoning steps that flow together, without losing any key information.

Reasoning steps should have a short title, followed a short description of the reasoning step.

You must ALWAYS include the final answer at the end of all the reasoning steps within \\boxed{{}}, exactly the same as it is given.

KEY POINTS:
- Do not miss any information from the summary. The final answer contains steps, but much less information than the summary. Add any reasoning steps from the summary where they belong
- Make the title of each reasoning move very short - ideally 5 words or less.
- Include all key information in the description of each resasoning step, including any mathematical steps.
- Do NOT make up math. If no math is present in a reasoning step, do not include it. If math is used, it should match the math in the reasoning step exactly.
- Do not refer to "the summary", "the final answer", "the solution above," or this summaries and final answers in any way — write entirely in your own voice, as an original and natural synthesis with no trace of a second source.

Format each reasoning step as follows, replacing the placeholders in <angle brackets> with content specific to that step. Never output the literal words "Title of reasoning move" or "Concise description" - those are placeholder labels, not example text:

- Step <number>: <a short, specific title for this step, in your own words>
    <description of this reasoning step, including any math if necessary>

Following the instructions above exactly, process the following summary and final answer:

SUMMARY: {summary}

FINAL ANSWER: {final_answer}

Again, you MUST include the final answer at the very end, within \\boxed{{}}. Title the bullet point 'Final Answer' and then provide the final answer within \\boxed{{}}.
Ensure that the final answer is formatted in the same way as and exactly matches the final answer in the provided final answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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

    # rejection sampling to remove explicit references to the summary provided to the expander
    for _ in range(NUM_SAMPLES):
        if len(remaining_samples) == 0:
            break

        failed_indices, offset = [], 0
        for batch in remaining_samples.iter(batch_size=batch_size):
            # get llm outputs
            outputs, correct_answers, _ = processing_function(batch, llm)

            # process and add to dataset
            for j, output in enumerate(outputs):
                if correct_answers[j]:
                    original_question = batch["question"][j]
                    original_summary = batch["summary"][j]
                    prompt = create_simple_prompt(original_question)
                    expansion = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                    expanded_dataset.append(
                        {
                            "prompt": prompt,
                            "question": original_question,
                            "completion": expansion,
                            "summary_trace": original_summary,
                            "num_tokens": num_tokens,
                        }
                    )
                else:
                    failed_indices.append(offset + j)
            offset += len(batch["question"])

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
    expand(batch_size=4096, processing_function=_expand_with_model)

"""
Generate a summarized dataset of reasoning traces
"""

import logging
import re
from typing import Any, Dict, Optional, List

from dataclasses import dataclass, field, asdict
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from vllm import LLM

from utils import _generate_batched_model_response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DefaultSummarizationConfiguration:
    """
    Default configurations for distillation on the model
    """

    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    dataset_path: str = field(default="treason/datasets/medium-14B")
    output_path: str = field(default="treason/datasets/medium-14B-summary-TEST")
    tensor_parallel_size: int = field(default=1)
    max_examples: Optional[int] = field(
        default=5,
        metadata={"help": "If set, only summarize the first N training rows"},
    )

def _process_dataset_batch(batch: Dict[str, Any], llm: LLM):
    """
    Summarize the reasoning traces in a batch of dataset examples
    """

    new_system_prompt = """
You are a concise summarizer, whose job is to take reasoning that is solving a problem, and simplify it into a conversational summary of the reasoning steps.

You must break down the reasoning process into a series of steps, each with a short title, followed by a textual description of the reasoning step. Include the final answer at the end within \\boxed{}.

Do not skip any steps - follow the same order as the original reasoning trace.
Make the title of each reasoning move very short - ideally 5 words or less.
Make the summary concise, but conversational and clear to the user what each step does.

Format each bullet point as follows, replacing the placeholders in <angle brackets> with content specific to that step. Never output the literal words "Title of reasoning move" or "Concise description" - those are placeholder labels, not example text:

'- <a short, specific title for this step, in your own words>
    <a concise, conversational description of what happens in this step>'

For example, here is one step in a reasoning trace and its corresponding summary: 

STEP IN REASONING TRACE: 
1. **Calculate the actual area of the circle:**
   The actual diameter of the circle is 20 cm, so the actual radius \( r \) is:
   \[
   r = \frac{20}{2} = 10 \text{ cm}
   \]
   The actual area \( A_{\text{actual}} \) of the circle is:
   \[
   A_{\text{actual}} = \pi r^2 = \pi (10)^2 = 100\pi \text{ cm}^2
   \]
SUMMARY:
'- Find the area of the circle 
    Find the radius of the circle, and multiply the radius squared by pi to get the area'

Make sure to include the final answer at the end within \\boxed{}. Title the bullet point 'Final Answer' and then provide the final answer within \\boxed{}.
Ensure that the final answer is formatted in the same way as and exactly matches the final answer in the original reasoning trace.

Now summarize the following using this format:
"""

    # get the prompts prompts the dataset
    messages = []
    for prompt in batch["completion"]:
        messages.append([{"role": "user", "content": prompt}])

    return _generate_batched_model_response(llm, new_system_prompt, messages, force_think=False)


def summarize(processing_function, batch_size=128):
    """
    Summarize the reasoning traces in a dataset
    """
    # get command line arguments
    parser = HfArgumentParser(DefaultSummarizationConfiguration)
    config = parser.parse_args_into_dataclasses()[0]
    logging.info(f"Summarize config: {asdict(config)}")

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

    # process dataset
    summarized_dataset: List[Dict[str, Any]] = []
    for batch in train_split.iter(batch_size=batch_size):
        # get llm outputs
        outputs = processing_function(batch, llm)
        # process and add to dataset
        for i, output in enumerate(outputs):
            original_prompt = batch["prompt"][i]
            original_completion = batch["completion"][i]
            # strip the start of the summary to prevent errors in prompt-completion loss calculation
            summary = output.outputs[0].text.lstrip()
            num_tokens = len(output.outputs[0].token_ids)
            summarized_dataset.append(
                {
                    "prompt": original_prompt,
                    "completion": summary,
                    "full_trace": original_completion,
                    "num_tokens": num_tokens,
                }
            )

    # save results
    summarized_train = Dataset.from_list(summarized_dataset)
    summarized_dataset = DatasetDict({"train": summarized_train})
    summarized_dataset.save_to_disk(config.output_path)
    logging.info(f"Saved summarized dataset to {config.output_path}")


if __name__ == "__main__":
    summarize(batch_size=4096, processing_function=_process_dataset_batch)

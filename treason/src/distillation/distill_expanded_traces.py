"""
Code for distillatoin. Inspired by code from Simple Test Time Scaling
https://github.com/simplescaling/s1/blob/main/train/sft.py
"""

import logging

from os import environ
from dataclasses import dataclass, field, asdict
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class DefaultDistillationConfiguration:
    """
    Default configurations for distillation on the model
    """

    model_name: str = field(default="Qwen/Qwen2.5-Math-7B")
    block_size: int = field(default=32768)
    training_dataset_path: str = field(default="treason/datasets/open-source-traces")
    wandb_project: str = field(default="distill-small")
    simplify_prompts: bool = field(default=False)

    def __post_init__(self):
        environ["WANDB_PROJECT"] = self.wandb_project


def distill_model():
    """
    Supervised fine-tuning of model on a distillation dataset
    """
    # parse the arguments from training
    parser = HfArgumentParser((DefaultDistillationConfiguration, SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info(f"Training config: {asdict(config), asdict(args)}")

    # load the model
    load_model_configs = {"torch_dtype": "auto", "attn_implementation": "sdpa", "use_cache": False}
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_model_configs)

    # load the dataset from the disk
    dataset = load_from_disk(config.training_dataset_path)
    
    # setting up the tokenizer and collator, to only learn from the answers
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        print("No padding token found. Setting pad token to be the eos token.")
        tokenizer.pad_token = tokenizer.eos_token

    # create SFT trainer
    args.max_seq_length = config.block_size
    args.completion_only_loss = True
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        args=args,
    )

    # train and save model
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    distill_model()

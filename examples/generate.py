#!/usr/bin/env python3

import argparse
import json
import sys
from typing import List

import torch
from curated_transformers.generation.config import (
    GreedyGeneratorConfig,
    SampleGeneratorConfig,
)
from curated_transformers.layers.attention import enable_torch_sdp
from curated_transformers.generation.auto_generator import AutoGenerator

EPILOG = """This program takes a JSON list of strings from the standard inputs as prompts. For example:

["What is spaCy?", "What is Rust?"]

The output is a JSON list with the following format:

[{"prompt": "What is spaCy?", "answer": "..."}, {"prompt": "What is Rust?", "answer": "..."}]'"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Generate from a list of prompts in JSON format",
    epilog=EPILOG,
)
parser.add_argument(
    "--device",
    "-d",
    metavar="DEVICE",
    default="cuda:0",
    help="device to place the model on (default: cuda:0)",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="tiiuae/falcon-7b-instruct",
    help="generation model (default: tiiuae/falcon-7b-instruct)",
)
parser.add_argument(
    "--temperature",
    "-t",
    metavar="T",
    type=float,
    default=1.0,
    help="softmax temperature (default: 1.0)",
)
parser.add_argument(
    "--top-k",
    "-k",
    metavar="K",
    type=int,
    default=0,
    help="top-k pieces to consider in sampling (default: disabled)",
)
parser.add_argument(
    "--sample",
    "-s",
    action="store_true",
    help="enable sampling",
)
parser.add_argument(
    "--torch-sdp",
    action="store_true",
    help="enable Torch scaled dot product attention implementation",
)


def read_prompts() -> List[str]:
    prompts_json = json.load(sys.stdin)
    if not isinstance(prompts_json, list):
        raise ValueError("Prompts must be a JSON list")

    prompts = []
    for prompt_json in prompts_json:
        if not isinstance(prompt_json, str):
            raise ValueError("Prompt must be a JSON string")
        prompts.append(prompt_json.strip())

    return prompts


if __name__ == "__main__":
    args = parser.parse_args()

    prompts = read_prompts()

    generator = AutoGenerator.from_hf_hub(
        name=args.model, device=torch.device(args.device)
    )

    config = (
        SampleGeneratorConfig(temperature=args.temperature, top_k=args.top_k)
        if args.sample
        else GreedyGeneratorConfig()
    )

    with enable_torch_sdp(args.torch_sdp):
        outputs = generator(prompts, config=config)

    output_dict = [
        {"prompt": prompt, "answer": answer} for prompt, answer in zip(prompts, outputs)
    ]
    print(json.dumps(output_dict))

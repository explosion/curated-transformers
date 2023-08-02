#!/usr/bin/env python3

# Rewrite sentences in the passive voice to the active voice using the
# 7B parameter Falcon model. This is only a small demo of using Curated
# Transformers for a specific application of generation. The example only
# works for relatively simple sentences, use larger models/better prompts/
# finetuning to get better results.
#
# Example usage:
#
# ./passive2active.py < passive-examples.json
#

import argparse
import json
import sys
from typing import List

import torch

from curated_transformers.generation.config import (
    GreedyGeneratorConfig,
    SampleGeneratorConfig,
)
from curated_transformers.generation.falcon import FalconGenerator
from curated_transformers.layers.attention import enable_torch_sdp
from curated_transformers.quantization.bnb import BitsAndBytesConfig

EPILOG = """This program takes passive sentences as a JSON list of strings
from the standard inputs. For example:

["The medal was won by the Dutch speed skater.", "Anita was driven to the theatre by Carla."]

The output is a JSON list with the following format:

[{"passive": "The medal was won by the Dutch speed skater.", "active": "..."},
 {"passive": "Anita was driven to the theatre by Carla.", "active": "..."}]'"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Rewrite sentences in the active voice to passive voice",
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
    "--quantize",
    "-q",
    metavar="B",
    type=int,
    choices=[4, 8],
    help="quantizate model to B bits",
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


def read_passives() -> List[str]:
    prompts_json = json.load(sys.stdin)
    if not isinstance(prompts_json, list):
        raise ValueError("Input must be a JSON list")

    prompts = []
    for prompt_json in prompts_json:
        if not isinstance(prompt_json, str):
            raise ValueError("Sentence in passive voice must be a JSON string")
        prompts.append(prompt_json.strip())

    return prompts


PROMPT = """Rewrite the following sentences in active voice:

Example 1:

Passive: "The vase was broken by the amateur skater."
Active: "The amateur skater broke the vase."

Example 2:

Passive: "The car is bought buy the man with the grey hat."
Active: "The man with the grey hat bought the car."

Rewrite this sentence in active voice:
Passive: "{passive}"
Active: "
"""


def preprocess_passives(passives: List[str]) -> List[str]:
    return [PROMPT.format(passive=passive) for passive in passives]


if __name__ == "__main__":
    args = parser.parse_args()

    quantization_config = None
    if args.quantize == 4:
        quantization_config = BitsAndBytesConfig.for_4bit()
    elif args.quantize == 8:
        quantization_config = BitsAndBytesConfig.for_8bit()

    passives = read_passives()

    generator = FalconGenerator.from_hf_hub(
        name=args.model,
        device=torch.device(args.device),
        quantization_config=quantization_config,
    )

    config = (
        SampleGeneratorConfig(temperature=args.temperature, top_k=args.top_k)
        if args.sample
        else GreedyGeneratorConfig()
    )

    with enable_torch_sdp(args.torch_sdp):
        actives = generator(preprocess_passives(passives), config=config)

    output_dict = [
        # Due to our prompting the model generates a closing quote.
        {"passive": passive, "active": active.rstrip('"')}
        for passive, active in zip(passives, actives)
    ]
    print(json.dumps(output_dict, indent=2))

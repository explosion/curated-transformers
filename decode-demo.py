#!/usr/bin/env python3

from curated_transformers.generation.models.dolly_v2 import DollyV2Generator

prompts = [
    "What is Python?",
    "What is spaCy?",
]

generator = DollyV2Generator.from_hf_hub(name="databricks/dolly-v2-3b")

pieces = [[] for _ in range(len(prompts))]
for step_pieces in generator(prompts):
    for seq_id, piece in step_pieces:
        pieces[seq_id].append(piece)

for prompt, answer in zip(prompts, pieces):
    print(f"Answer to: {prompt}")
    print("".join(answer), end=None)

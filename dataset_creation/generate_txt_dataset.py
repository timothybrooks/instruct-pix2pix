from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import openai
from tqdm.auto import tqdm


DELIMITER_0 = "\n##\n"
DELIMITER_1 = "\n%%\n"
STOP = "\nEND"


def generate(
    openai_model: str,
    caption: str,
    num_retries: int = 3,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.1,
    presence_penalty: float = 0.0,
    sleep_on_error: float = 1.0,
) -> Optional[tuple[str, str]]:
    for _ in range(1 + num_retries):
        try:
            response = openai.Completion.create(
                model=openai_model,
                prompt=caption + DELIMITER_0,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=[STOP],
            )
        except Exception as e:
            print(e)
            time.sleep(sleep_on_error)
            continue
        output = response["choices"][0]["text"].split(DELIMITER_1)
        if len(output) == 2:
            instruction, edited_caption = output
            results = openai.Moderation.create([instruction, edited_caption])["results"]
            if results[0]["flagged"] or results[1]["flagged"]:
                continue
            if caption.strip().strip(".!?").lower() != edited_caption.strip().strip(".!?").lower():
                return instruction, edited_caption


def main(openai_model: str, num_samples: int, num_partitions: int, partition: int, seed: int):
    dataset = datasets.load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus", split="train")
    # Other datasets we considered that may be worth trying:
    # dataset = datasets.load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train")
    # dataset = datasets.load_dataset("laion/laion-coco", split="train")

    np.random.seed(seed)
    permutation = np.array_split(np.random.permutation(len(dataset)), num_partitions)[partition]
    dataset = dataset[permutation]
    captions = dataset["TEXT"]
    urls = dataset["URL"]
    output_path = f"data/dataset=laion-aesthetics-6.5_model={openai_model}_samples={num_samples}_partition={partition}.jsonl"  # fmt: skip
    print(f"Prompt file path: {output_path}")

    count = 0
    caption_set = set()
    url_set = set()

    if Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in tqdm(f, desc="Resuming from existing prompts"):
                prompt = json.loads(line)
                if prompt["caption"] not in caption_set and prompt["url"] not in url_set:
                    caption_set.add(prompt["caption"])
                    url_set.add(prompt["url"])
                    count += 1

    with open(output_path, "a") as fp:
        with tqdm(total=num_samples - count, desc="Generating instructions and edited captions") as progress_bar:
            for caption, url in zip(captions, urls):
                if caption in caption_set or url in url_set:
                    continue
                if openai.Moderation.create(caption)["results"][0]["flagged"]:
                    continue
                edit_output = generate(openai_model, caption)
                if edit_output is not None:
                    edit, output = edit_output
                    fp.write(f"{json.dumps(dict(caption=caption, edit=edit, output=output, url=url))}\n")
                    count += 1
                    progress_bar.update()
                    caption_set.add(caption)
                    url_set.add(url)
                if count == num_samples:
                    break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--openai-api-key", required=True, type=str)
    parser.add_argument("--openai-model", required=True, type=str)
    parser.add_argument("--num-samples", default=10000, type=int)
    parser.add_argument("--num-partitions", default=1, type=int)
    parser.add_argument("--partition", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    openai.api_key = args.openai_api_key
    main(args.openai_model, args.num_samples, args.num_partitions, args.partition, args.seed)

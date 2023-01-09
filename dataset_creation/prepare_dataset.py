import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    seeds = []
    with tqdm(desc="Listing dataset image seeds") as progress_bar:
        for prompt_dir in dataset_dir.iterdir():
            if prompt_dir.is_dir():
                prompt_seeds = [image_path.name.split("_")[0] for image_path in sorted(prompt_dir.glob("*_0.jpg"))]
                if len(prompt_seeds) > 0:
                    seeds.append((prompt_dir.name, prompt_seeds))
                    progress_bar.update()
    seeds.sort()

    with open(dataset_dir.joinpath("seeds.json"), "w") as f:
        json.dump(seeds, f)


if __name__ == "__main__":
    main()

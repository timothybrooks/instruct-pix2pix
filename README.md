# InstructPix2Pix: Learning to Follow Image Editing Instructions
### [Project Page](https://www.timothybrooks.com/instruct-pix2pix/) | [Paper](https://arxiv.org/abs/2211.09800) | [Data](http://instruct-pix2pix.eecs.berkeley.edu/)
PyTorch implementation of InstructPix2Pix, an instruction-based image editing model, based on the original [CompVis/stable_diffusion](https://github.com/CompVis/stable-diffusion) repo. <br>

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix/)  
 [Tim Brooks](https://www.timothybrooks.com/)\*,
 [Aleksander Holynski](https://holynski.org/)\*,
 [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/) <br>
 UC Berkeley <br>
  \*denotes equal contribution  
  
  <img src='https://instruct-pix2pix.timothybrooks.com/teaser.jpg'/>

## TL;DR: quickstart 

To setup a conda environment, download a pretrained model, and edit an image:
```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"

# Optionally, you can specify parameters:
# python edit_cli.py --steps 100 --resolution 512 --seed 0 --cfg-text 7.5 --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"
```

## Setup

Install all dependencies with:
```
conda env create -f environment.yaml
```

Download the pretrained models by running:
```
bash scripts/download_checkpoints.sh
```

## Generated Dataset

Our image editing model is trained on a generated dataset consisting of 454,445 examples. Each example contains (1) an input image, (2) an editing instruction, and (3) an output edited image. We provide two versions of the dataset, one in which each pair of edited images is generated 100 times, and the best examples are chosen based on CLIP metrics (Section 3.1.2 in the paper) (`clip-filtered-dataset`), and one in which examples are randomly chosen (`random-sample-dataset`).

For the released version of this dataset, we've additionally filtered prompts and images for NSFW content. After NSFW filtering, the GPT-3 generated dataset contains 451,990 examples. The final image-pair datasets contain:

|  | # of image editing examples | Dataset size |
|--|-----------------------|----------------------- |
| `random-sample-dataset` |451990|727GB|
|  `clip-filtered-dataset` |313010|436GB|

To download one of these datasets, along with the entire NSFW-filtered text data, run the following command with the appropriate dataset name:

```
bash scripts/download_data.sh clip-filtered-dataset
```


## Training InstructPix2Pix

Need to modify configs/instruct-pix2pix/default.yaml to point to the dataset in the right location. Need to also download the Stable Diffusion checkpoint from which to finetune.

```
python stable_diffusion/main.py --name default --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```


## Creating your own dataset

Our generated dataset of paired images and editing instructions is made in two phases: First, we use GPT-3 to generate text triplets: (a) a caption describing an image, (b) an edit instruction, (c) a caption describing the image after the edit. Then, we turn pairs of captions (before/after the edit) into pairs of images using Stable Diffusion and Prompt-to-Prompt.

### (1) Generate a dataset of captions and instructions

We provide our generated dataset of captions and edit instructions [here](https://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl). If you plan to use our captions+instructions, skip to step (2). Otherwise, if you would like to create your own text dataset, please follow steps (1.1-1.3) below. Note that generating very large datasets using GPT-3 can be expensive.

#### (1.1) Manually write a dataset of instructions and captions

The first step of the process is fine-tuning GPT-3. To do this, we made a dataset of 700 examples broadly covering of edits that we might want our model to be able to perform. Our examples are available here [here](https://instruct-pix2pix.eecs.berkeley.edu/human_written_examples.jsonl). These should be diverse and cover a wide range of possible captions and types of edits. Ideally, they should avoid duplication or significant overlap of captions and instructions. It is also important to be mindful of limitations of Stable Diffusion and Prompt-to-Prompt in writing these examples, such as inability to perform large spatial transformations (e.g., moving the camera, zooming in, swapping object locations). 

Input prompts should closely match the distribution of input prompts used to generate the larger dataset. We sampled the 700 input prompts from LAION Improves Aesthetics 6.5+ dataset and also use this dataset for generating examples. We found this dataset is quite noisy (many of the captions are overly long and contain irrelevant text). For this reason, we also considered MSCOCO and LAION-COCO datasets, but ultimately chose LAION Improves Aesthetics 6.5+ due to its diversity of content, proper nouns, and artistic mediums. If you choose to use another dataset or combination of datasets as input to GPT-3 when generating examples, we recomend you sample the input prompts from the same distribution when manually writing training examples.

#### (1.2) Finetune GPT-3

The next step is to finetune a large language model to generate an edit instruction and edited caption from a new input caption. We use GPT-3 Davinci via the OpenAI API, although other language models could be used.

To prepare training data for GPT-3, one must setup an OpenAI developer account to access the needed APIs. Run the `prompts/prepare_for_gpt.py` script, which forms the prompts into the correct format by concatenating instructions and captions and adding delimiters and stop sequences.

```bash
python dataset_creation/prepare_for_gpt.py prompts/human_written_examples.jsonl prompts/human_written_examples_for_gpt.jsonl
```

Next, finetune GPT-3 via the OpenAI CLI. We provide an example below, although please refer to the official documentation here as best practices may change. We trained the Davinci model for a single epoch. You could experiment with smaller less expensive GPT-3 variants or with open source language models, although this may negatively hurt performance.

```bash
openai api fine_tunes.create -t prompts/human_written_examples_for_gpt.jsonl -m davinci --n_epochs 1 --suffix "instruct-pix2pix"
```

You can test out the finetuned GPT-3 model by launching the provided Gradio app:

```bash
python prompt_app.py OPENAI_MODEL_NAME
```

#### (1.3) Generate a large dataset of captions and instructions

We now use the finetuned GPT-3 model to generate a large dataset. Our dataset cost thousands of dollars to create. See `prompts/gen_instructions_and_captions.py` for the script which generates these examples. We recommend first generating a small number of examples and gradually increasing the scale to ensure the results are working as desired before increasing scale.

```bash
python dataset_creation/generate_txt_dataset.py OPENAI_MODEL_NAME
```

If you are generating at a very large scale (e.g., 100K+), it will be noteably faster to generate the dataset with multiple processes running in parallel. This can be accomplished by setting `--partitions=N` to a higher number and running multiple processes, setting each `--partition` to the corresponding value.

```bash
python dataset_creation/generate_txt_dataset.py OPENAI_MODEL_NAME --partitions=10 --partition=0
```

### (2) Turn paired captions into paired images

The next step is to turn pairs of text captions into pairs of images. For this, we need to copy a pre-trained Stable Diffusion model checkpoint to `stable_diffusion/models/ldm/stable-diffusion-v1/`. For our model, we used [checkpoint v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt), but other versions may also work. It is also necessary to download a checkpoint for the Stable Diffusion autoencoder. We used the [new autoencoder](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt), which should be put in the same directory. Once all checkpoints have been downloaded, we can generate the dataset with the following command:

```
python dataset_creation/generate_img_dataset.py data/instruct-pix2pix-dataset-000 data/gpt_generated_prompts.jsonl
```

This command operates on a single GPU (typically a V100 or A100). To parallelize over many GPUs/machines, set `--n-partitions` to the total number of parallel jobs and `--partition` to the index of each job.

```
python dataset_creation/generate_img_dataset.py data/instruct-pix2pix-dataset-000 data/gpt_generated_prompts.jsonl --n-partitions 100 --partition 0
```

The default parameters match that of our dataset, although in practice you can use a smaller number of steps (e.g., `--steps=25`) to generate high quality data faster. By default, we generate 100 samples per prompt and use CLIP filtering to keep a max of 4 per prompt. You can experiment with fewer samples by setting `--n-samples`. The command below turns off CLIP filtering entirely and is therefore faster:

```
python dataset_creation/generate_img_dataset.py data/instruct-pix2pix-dataset-000 data/gpt_generated_prompts.jsonl --n-samples 4 --clip-threshold 0 --clip-dir-threshold 0 --clip-img-threshold 0 --n-partitions 100 --partition 0
```

After generating all of the dataset examples, run the following command below to create a list of the examples. This is needed for the dataset onject to efficiently be able to sample examples without needing to iterate over the entire dataset directory at the start of each training run.

```
python dataset_creation/prepare_dataset.py data/instruct-pix2pix-dataset-000
```

## Comments

- Our codebase is based on the [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion).

## BibTeX

```
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```




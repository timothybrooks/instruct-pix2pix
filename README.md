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

Follow the instructions below to download and run InstructPix2Pix on your own images. These instructions have been tested on a GPU with >18GB VRAM. If you don't have a GPU, you may need to change the default configuration, or check out [other ways of using the model](https://github.com/timothybrooks/instruct-pix2pix#other-ways-of-using-instructpix2pix). 

### Set up a conda environment, and download a pretrained model:
```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```

### Edit a single image:
```
python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"

# Optionally, you can specify parameters to tune your result:
# python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit "turn him into a cyborg"
```

### Or launch your own interactive editing Gradio app:
```
python edit_app.py 
```
![Edit app](https://github.com/timothybrooks/instruct-pix2pix/blob/main/imgs/edit_app.jpg?raw=true)

_(For advice on how to get the best results by tuning parameters, see the [Tips](https://github.com/timothybrooks/instruct-pix2pix#tips) section)._

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

InstructPix2Pix is trained by fine-tuning from an initial StableDiffusion checkpoint. The first step is to download a Stable Diffusion checkpoint. For our trained models, we used the v1.5 checkpoint as the starting point. To download the same ones we used, you can run the following script:
```
bash scripts/download_pretrained_sd.sh
```
If you'd like to use a different checkpoint, point to it in the config file `configs/train.yaml`, on line 8, after `ckpt_path:`. 

Next, we need to change the config to point to our downloaded (or generated) dataset. If you're using the `clip-filtered-dataset` from above, you can skip this. Otherwise, you may need to edit lines 85 and 94 of the config (`data.params.train.params.path`, `data.params.validation.params.path`). 

Finally, start a training job with the following command:

```
python main.py --name default --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```


## Creating your own dataset

Our generated dataset of paired images and editing instructions is made in two phases: First, we use GPT-3 to generate text triplets: (a) a caption describing an image, (b) an edit instruction, (c) a caption describing the image after the edit. Then, we turn pairs of captions (before/after the edit) into pairs of images using Stable Diffusion and Prompt-to-Prompt.

### (1) Generate a dataset of captions and instructions

We provide our generated dataset of captions and edit instructions [here](https://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl). If you plan to use our captions+instructions, skip to step (2). Otherwise, if you would like to create your own text dataset, please follow steps (1.1-1.3) below. Note that generating very large datasets using GPT-3 can be expensive.

#### (1.1) Manually write a dataset of instructions and captions

The first step of the process is fine-tuning GPT-3. To do this, we made a dataset of 700 examples broadly covering of edits that we might want our model to be able to perform. Our examples are available [here](https://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl). These should be diverse and cover a wide range of possible captions and types of edits. Ideally, they should avoid duplication or significant overlap of captions and instructions. It is also important to be mindful of limitations of Stable Diffusion and Prompt-to-Prompt in writing these examples, such as inability to perform large spatial transformations (e.g., moving the camera, zooming in, swapping object locations). 

Input prompts should closely match the distribution of input prompts used to generate the larger dataset. We sampled the 700 input prompts from the _LAION Improved Aesthetics 6.5+_ dataset and also use this dataset for generating examples. We found this dataset is quite noisy (many of the captions are overly long and contain irrelevant text). For this reason, we also considered MSCOCO and LAION-COCO datasets, but ultimately chose _LAION Improved Aesthetics 6.5+_ due to its diversity of content, proper nouns, and artistic mediums. If you choose to use another dataset or combination of datasets as input to GPT-3 when generating examples, we recommend you sample the input prompts from the same distribution when manually writing training examples.

#### (1.2) Finetune GPT-3

The next step is to finetune a large language model on the manually written instructions/outputs to generate edit instructions and edited caption from a new input caption. For this, we finetune GPT-3's Davinci model via the OpenAI API, although other language models could be used.

To prepare training data for GPT-3, one must first create an OpenAI developer account to access the needed APIs, and [set up the API keys on your local device](https://beta.openai.com/docs/api-reference/introduction). Also, run the `prompts/prepare_for_gpt.py` script, which forms the prompts into the correct format by concatenating instructions and captions and adding delimiters and stop sequences.

```bash
python dataset_creation/prepare_for_gpt.py --input-path data/human-written-prompts.jsonl --output-path data/human-written-prompts-for-gpt.jsonl
```

Next, finetune GPT-3 via the OpenAI CLI. We provide an example below, although please refer to OpenAI's official documentation for this, as best practices may change. We trained the Davinci model for a single epoch. You can experiment with smaller less expensive GPT-3 variants or with open source language models, although this may negatively affect performance.

```bash
openai api fine_tunes.create -t data/human-written-prompts-for-gpt.jsonl -m davinci --n_epochs 1 --suffix "instruct-pix2pix"
```

You can test out the finetuned GPT-3 model by launching the provided Gradio app:

```bash
python prompt_app.py --openai-api-key OPENAI_KEY --openai-model OPENAI_MODEL_NAME
```

![Prompt app](https://github.com/timothybrooks/instruct-pix2pix/blob/main/imgs/prompt_app.jpg?raw=true)

#### (1.3) Generate a large dataset of captions and instructions

We now use the finetuned GPT-3 model to generate a large dataset. Our dataset cost thousands of dollars to create. See `prompts/gen_instructions_and_captions.py` for the script which generates these examples. We recommend first generating a small number of examples (by setting a low value of `--num-samples`) and gradually increasing the scale to ensure the results are working as desired before increasing scale.

```bash
python dataset_creation/generate_txt_dataset.py --openai-api-key OPENAI_KEY --openai-model OPENAI_MODEL_NAME
```

If you are generating at a very large scale (e.g., 100K+), it will be noteably faster to generate the dataset with multiple processes running in parallel. This can be accomplished by setting `--partitions=N` to a higher number and running multiple processes, setting each `--partition` to the corresponding value.

```bash
python dataset_creation/generate_txt_dataset.py --openai-api-key OPENAI_KEY --openai-model OPENAI_MODEL_NAME --partitions=10 --partition=0
```

### (2) Turn paired captions into paired images

The next step is to turn pairs of text captions into pairs of images. For this, we need to copy some pre-trained Stable Diffusion checkpoints to `stable_diffusion/models/ldm/stable-diffusion-v1/`. You may have already done this if you followed the instructions above for training with our provided data, but if not, you can do this by running:

```bash
bash scripts/download_pretrained_sd.sh
```

For our model, we used [checkpoint v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt), and the [new autoencoder](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt), but other models may work as well. If you choose to use other models, make sure to change point to the corresponding checkpoints by passing in the `--ckpt` and `--vae-ckpt` arguments. Once all checkpoints have been downloaded, we can generate the dataset with the following command:

```
python dataset_creation/generate_img_dataset.py --out_dir data/instruct-pix2pix-dataset-000 --prompts_file path/to/generated_prompts.jsonl
```

This command operates on a single GPU (typically a V100 or A100). To parallelize over many GPUs/machines, set `--n-partitions` to the total number of parallel jobs and `--partition` to the index of each job.

```
python dataset_creation/generate_img_dataset.py --out_dir data/instruct-pix2pix-dataset-000 --prompts_file path/to/generated_prompts.jsonl --n-partitions 100 --partition 0
```

The default parameters match that of our dataset, although in practice you can use a smaller number of steps (e.g., `--steps=25`) to generate high quality data faster. By default, we generate 100 samples per prompt and use CLIP filtering to keep a max of 4 per prompt. You can experiment with fewer samples by setting `--n-samples`. The command below turns off CLIP filtering entirely and is therefore faster:

```
python dataset_creation/generate_img_dataset.py --out_dir data/instruct-pix2pix-dataset-000 --prompts_file path/to/generated_prompts.jsonl --n-samples 4 --clip-threshold 0 --clip-dir-threshold 0 --clip-img-threshold 0 --n-partitions 100 --partition 0
```

After generating all of the dataset examples, run the following command below to create a list of the examples. This is needed for the dataset onject to efficiently be able to sample examples without needing to iterate over the entire dataset directory at the start of each training run.

```
python dataset_creation/prepare_dataset.py data/instruct-pix2pix-dataset-000
```

## Evaluation

To generate plots like the ones in Figures 8 and 10 in the paper, run the following command:

```
python metrics/compute_metrics.py --ckpt /path/to/your/model.ckpt
```

## Tips

If you're not getting the quality result you want, there may be a few reasons:
1. **Is the image not changing enough?** Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Increasing the Text CFG weight, or
2. Conversely, **is the image changing too much**, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try cropping the image so the face takes up a larger portion of the frame.

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
## Other ways of using InstructPix2Pix

### InstructPix2Pix on [HuggingFace](https://huggingface.co/spaces/timbrooks/instruct-pix2pix):
> A browser-based version of the demo is available as a [HuggingFace space](https://huggingface.co/spaces/timbrooks/instruct-pix2pix). For this version, you only need a browser, a picture you want to edit, and an instruction! Note that this is a shared online demo, and processing time may be slower during peak utilization. 

### InstructPix2Pix on [Replicate](https://replicate.com/timothybrooks/instruct-pix2pix):
> Replicate provides a production-ready cloud API for running the InstructPix2Pix model. You can run the model from any environment using a simple API call with cURL, Python, JavaScript, or your language of choice. Replicate also provides a web interface for running the model and sharing predictions.

### InstructPix2Pix in [Imaginairy](https://github.com/brycedrennan/imaginAIry#-edit-images-with-instructions-alone-by-instructpix2pix):
> Imaginairy offers another way of easily installing InstructPix2Pix with a single command. It can run on devices without GPUs (like a Macbook!). 
> ```bash
> pip install imaginairy --upgrade
> aimg edit any-image.jpg --gif "turn him into a cyborg" 
> ```
> It also offers an easy way to perform a bunch of edits on an image, and can save edits out to an animated GIF:
> ```
> aimg edit --gif --surprise-me pearl-earring.jpg 
> ```
> <img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/7c05c3aae2740278978c5e84962b826e58201bac/assets/girl_with_a_pearl_earring_suprise.gif" width="512">

### InstructPix2Pix in [🧨 Diffusers](https://github.com/huggingface/diffusers):

> InstructPix2Pix in Diffusers is a bit more optimized, so it may be faster and more suitable for GPUs with less memory. Below are instructions for installing the library and editing an image: 
> 1. Install diffusers and relevant dependencies:
>
> ```bash
> pip install transformers accelerate torch
>
> pip install git+https://github.com/huggingface/diffusers.git
> ```
> 
> 2. Load the model and edit the image:
>
> ```python
> 
> import torch
> from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
> 
> model_id = "timbrooks/instruct-pix2pix"
> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
> pipe.to("cuda")
> pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
> # `image` is an RGB PIL.Image
> images = pipe("turn him into cyborg", image=image).images
> images[0]
> ```
> 
> For more information, check the docs [here](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/pix2pix).

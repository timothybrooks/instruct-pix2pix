#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../stable_diffusion/models/ldm/stable-diffusion-v1
curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o $SCRIPT_DIR/../stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt
curl -L https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -o $SCRIPT_DIR/../stable_diffusion/models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt

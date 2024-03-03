#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR="${SCRIPT_DIR}/../stable_diffusion/models/ldm/stable-diffusion-v1"

mkdir -p ${DATA_DIR}

if ( [ $HF_HUB_ENABLE_HF_TRANSFER ] ); then
	echo "got into if"
	# assumes that huggingface-cli and hf-transfer are installed and HF_HUB_ENABLE_HF_TRANSFER=1
	# in environment globals. It also requires you have a HF_TOKEN set.
	# NB although faster, hf-transfer doesn't support debugging or resuming partial downloads
	# see https://huggingface.co/docs/huggingface_hub/en/guides/download#download-from-the-cli
	huggingface-cli download --local-dir $DATA_DIR "runwayml/stable-diffusion-v1-5" v1-5-pruned-emaonly.ckpt
	huggingface-cli download --local-dir $DATA_DIR "stabilityai/sd-vae-ft-mse-original" vae-ft-mse-840000-ema-pruned.ckpt
else
	echo "got into else"
	curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -o $DATA_DIR/v1-5-pruned-emaonly.ckpt
	curl -L https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -o $DATA_DIR/vae-ft-mse-840000-ema-pruned.ckpt
fi 


#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../checkpoints

wget -t15 -c --progress=bar -w15 --retry-connrefused http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt -O $SCRIPT_DIR/../checkpoints/instruct-pix2pix-00-22000.ckpt

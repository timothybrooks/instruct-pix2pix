#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../checkpoints

curl http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt -o $SCRIPT_DIR/../checkpoints/instruct-pix2pix-00-22000.ckpt

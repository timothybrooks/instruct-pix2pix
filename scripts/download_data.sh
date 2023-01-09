#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data

wget http://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl -O $SCRIPT_DIR/../data/gpt-generated-prompts.jsonl
wget http://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl -O $SCRIPT_DIR/../data/human-written-prompts.jsonl

mkdir $SCRIPT_DIR/../data/$1
wget -A zip,json -r http://instruct-pix2pix.eecs.berkeley.edu/$1 -nd -P $SCRIPT_DIR/../data/$1

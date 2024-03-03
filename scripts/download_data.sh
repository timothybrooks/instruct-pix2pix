#!/bin/bash

# Make data folder relative to script location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../data

# Copy text datasets
wget -nc -c -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/gpt-generated-prompts.jsonl -O $SCRIPT_DIR/../data/gpt-generated-prompts.jsonl
wget -nc -c -q --show-progress http://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl -O $SCRIPT_DIR/../data/human-written-prompts.jsonl

# If dataset name isn't provided, exit. 
if [ -z $1 ] 
then 
	exit 0 
fi

# Copy dataset files
mkdir $SCRIPT_DIR/../data/$1
wget -nc -A zip,json -R "index.html*" -q --show-progress -r --no-parent -c http://instruct-pix2pix.eecs.berkeley.edu/$1/ -nd -P $SCRIPT_DIR/../data/$1/

if [ -z $2 -a ( $2="master" )]; then
	True
else
	exit 2
fi

# Unzip to folders
unzip $SCRIPT_DIR/../data/$1/\*.zip -d $SCRIPT_DIR/../data/$1/

# Cleanup
rm -f $SCRIPT_DIR/../data/$1/*.zip
rm -f $SCRIPT_DIR/../data/$1/*.html

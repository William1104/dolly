#!/bin/env bash

transformers-cli download EleutherAI/gpt-j-6B --cache-dir ./model/
export TRANSFORMERS_CACHE=`pwd`/model

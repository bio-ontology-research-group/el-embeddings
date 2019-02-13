#!/bin/bash
for i in $(seq 0 1 29)
do
python elembedding.py -pai $i -e 10
done

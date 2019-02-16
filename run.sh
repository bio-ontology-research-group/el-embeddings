#!/bin/bash
for i in $(seq 15 1 19)
do
python evaluate_interactions.py -pai $i > $i.res
done

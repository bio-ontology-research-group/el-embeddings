#!/bin/bash
for i in $(seq 0 1 100)
do
python plot_embeddings.py -cef data/cls_embeddings.pkl_$i.pkl -ref data/rel_embeddings.pkl_$i.pkl -e $i
done

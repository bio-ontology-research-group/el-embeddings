#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
from collections import deque

from utils import Ontology, FUNC_DICT

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/data-test/4932.protein.actions.v11.txt',
    help='')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/data-train/yeast_cls_embeddings.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/data-train/yeast_rel_embeddings.pkl',
    help='Relation embedings file')
def main(go_file, data_file, cls_embeds_file, rel_embeds_file):
    go = Ontology(go_file, with_rels=False)

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)
    embeds_list = cls_df['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {v: k for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    rs = np.abs(embeds[:, -1]).reshape(-1, 1)
    embeds = embeds[:, :-1]

    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb
    
    data = load_data(data_file, classes, relations)
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    n = len(data)
    margin = 0.01
    with ck.progressbar(data) as prog_data:
        for c, r, d in prog_data:
            c, r, d = classes[c], relations[r], classes[d]
            ec = embeds[c, :]
            rc = rs[c, :]
            er = rembeds[r, :]
            ec += er

            dst = np.linalg.norm(embeds - ec.reshape(1, -1), axis=1)
            dst = dst.reshape(-1, 1)
            overlap = np.maximum(0, (2 * rc - np.maximum(dst + rc - rs - margin, 0)) / (2 * rc))
            edst = np.maximum(0, dst - rc - rs - margin)
            res = (overlap + 1 / np.exp(edst)) / 2
            res = res.flatten()
            index = np.argsort(res)[::-1]
            rank = 1
            for i, nd in enumerate(index):
                if nd == d:
                    break
                if res[nd] != 1.0:
                    rank += 1
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
        print(top1, top10, top100, mean_rank)
        print(top1 / n, top10 / n, top100 / n, mean_rank / n)
    
def load_data(data_file, classes, relations):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            rel = f'<http://{it[2]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((id1, rel, id2))
    return data

def is_inside(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst + rc <= rd

def is_intersect(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst <= rc + rd
    
def sim(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    overlap = max(0, (2 * rc - max(dst + rc - rd, 0)) / (2 * rc))
    edst = max(0, dst - rc - rd)
    res = (overlap + 1 / np.exp(edst)) / 2


    
if __name__ == '__main__':
    main()

#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
from collections import deque

from utils import Ontology, FUNC_DICT
from elembedding import load_data

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/data-train/yeast-classes-normalized.owl',
    help='Data file')
@ck.option(
    '--neg-data-file', '-ndf', default='data/go-negatives.txt',
    help='Negative subclass relations (generate_negatives.py)')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/cls_embeddings.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/rel_embeddings.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=0.01,
    help='Margin parameter used for training')
def main(go_file, data_file, neg_data_file, cls_embeds_file, rel_embeds_file, margin):
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
    rs = np.abs(embeds[:, -1])
    embeds = embeds[:, :-1]

    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb
    
    data, _, _, _ = load_data(data_file, neg_data_file, index=False)
    
    print(relations)
    # Evaluate normal form 1 axioms
    n = 0
    s = 0
    for c, d in data['nf1']:
        if c not in classes or d not in classes:
            continue
        n += 1
        c, d = classes[c], classes[d]
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]
        if is_inside(ec, rc, ed, rd, margin):
            s += 1
    print('Normal form 1', n, s, s / n)

    # Normal form 2 axioms
    n = 0
    s = 0
    ns = 0
    for c, d, e in data['nf2']:
        if c not in classes or d not in classes or e not in classes:
            continue
        n += 1
        c, d, e = classes[c], classes[d], classes[e]
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]
        ee = embeds[e, :]
        re = rs[e]
        dst = np.linalg.norm(ec - ed) - margin
        if dst <= rc + rd and (is_inside(ec, rc, ee, re, margin) or is_inside(ed, rd, ee, re, margin)):
            s += 1
        elif (dst > rc and dst > rd) and dst <= rc + rd:
            x = (dst * dst - rc * rc + rd * rd) / (2 * dst)
            rx = math.sqrt(rd * rd - x * x)
            c = x / dst
            ex = ed + (ec - ed) * c
            if is_inside(ex, rx, ee, re, margin):
                s += 1
        elif dst > rc + rd:
            ns += 1    
        
    print('Normal form 2', n, s, s / n, ns)

    # Evaluate normal form 3 axioms
    # C subclassOf R some D
    n = 0 # len(data['nf3'])
    s = 0
    for c, r, d in data['nf3']:
        if c not in classes or d not in classes or r not in relations:
            continue
        c, r, d = classes[c], relations[r], classes[d]
        if r not in [0, 1, 3, 7, 9, 10, 15]:
            continue
        n += 1
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]
        er = rembeds[r, :]
        ec = ec + er
        if is_inside(ec, rc, ed, rd, margin):
            s += 1
    print('Normal form 3', n, s, s / n)

    # Evaluate normal form 4 axioms
    # R some C subclassOf D
    n = 0
    s = 0
    for r, c, d in data['nf4']:
        if c not in classes or d not in classes or r not in relations:
            continue
        n += 1
        r, c, d = relations[r], classes[c], classes[d]
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]
        er = rembeds[r, :]
        ec = ec - er
        if is_intersect(ec, rc, ed, rd, margin):
            s += 1
    print('Normal form 4', n, s, s / n)

    # Disjointness axioms
    n = len(data['disjoint'])
    s = 0
    for c, d, e in data['disjoint']:
        c, d = classes[c], classes[d]
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]

        if not is_intersect(ec, rc, ed, rd):
            s += 1
    print('Disjointness', n, s, s / n)

    # plot_embeddings(embeds, rs, classes)
    return
    g = {}
    for i in range(len(embeds)):
        g[i] = []
    for c, d in data['nf1']:
        g[d].append(c)

    sub_n = 1000
    labels = np.zeros((sub_n, len(embeds)), dtype=np.int8)
    
    print('Building labels')
    
    for i in range(sub_n):
        q = deque()
        for ch in g[i]:
            q.append(ch)
        while len(q) > 0:
            c = q.popleft()
            for ch in g[c]:
                q.append(ch)
            labels[i, c] = 1

    print('Running inference')
    preds = np.zeros((sub_n, len(embeds)), dtype=np.int8)
    for i in range(sub_n):
        c = embeds[i, :]
        rc = rs[i]
        dst = np.linalg.norm(embeds - c, axis=1)
        dst = dst + rs - margin
        subs = (dst <= rc).astype(np.int8)
        preds[i, :] = subs

    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f = 2 * precision * recall / ( precision + recall)
    print(f, precision, recall)


def is_inside(ec, rc, ed, rd, margin=0.01):
    dst = np.linalg.norm(ec - ed)
    return dst + rc - rd - margin <= 0

def is_intersect(ec, rc, ed, rd, margin=0.01):
    dst = np.linalg.norm(ec - ed)
    return dst - margin <= rc + rd

def sim(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    overlap = max(0, (2 * rc - max(dst + rc - rd, 0)) / (2 * rc))
    edst = max(0, dst - rc - rd)
    res = (overlap + 1 / np.exp(edst)) / 2


    
if __name__ == '__main__':
    main()

#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
from collections import deque

from utils import Ontology, FUNC_DICT

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/data-test/4932.protein.actions.v11.txt',
    help='')
@ck.option(
    '--sim-score-file', '-ssf', default='data/sim_proteins_yeast.txt',
    help='Semantic similarity scores for protein pairs (SemanticSimilarity.groovy)')
def main(go_file, data_file, sim_score_file):
    go = Ontology(go_file, with_rels=False)

    with open(sim_score_file, 'r') as f:
        proteins = next(f).strip().split('\t')
        prots_dict = {v: k for k, v in enumerate(proteins)}
        sim = np.zeros((len(proteins), len(proteins)), dtype=np.float32)
        i = 0
        for line in f:
            line = line.replace('null', '0.0')
            s = line.strip().split('\t')
            s = np.array(list(map(float, s)), dtype=np.float32)
            sim[i, :] = s
            i += 1
        
    data = load_data(data_file, prots_dict)
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    n = len(data)
    labels = np.zeros((len(proteins), len(proteins)), dtype=np.int32) 
    
    with ck.progressbar(data) as prog_data:
        for c, d in prog_data:
            labels[c, d] = 1
            labels[d, c] = 1
            index = rankdata(-sim[c, :], method='average')
            rank = index[d]
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
        print()
        print(top1 / n, top10 / n, top100 / n, mean_rank / n)
    # roc_auc = compute_roc(l, p)
    # fmax = compute_fmax(l, p)
    # print(rel_df['relations'][i], roc_auc)
    # print(fmax)
    print('Global', compute_roc(labels, sim))
    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0.0
    rmax = 0.0
    tmax = 0
    tpmax = 0
    fpmax = 0
    fnmax = 0
    for t in range(101):
        th = t / 100
        predictions = (preds >= th).astype(np.int32)
        tp = np.sum(labels & predictions)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f = 2 * (p * r) / (p + r)
        if f > fmax:
            fmax = f
            pmax = p
            rmax = r
            tmax = t
            tpmax, fpmax, fnmax = tp, fp, fn
    return fmax, pmax, rmax, tmax, tpmax, fpmax, fnmax
    
def load_data(data_file, proteins):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            rel = it[2]
            if id1 not in proteins or id2 not in proteins:
                continue
            data.append((proteins[id1], proteins[id2]))
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

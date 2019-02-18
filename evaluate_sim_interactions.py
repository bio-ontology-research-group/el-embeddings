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
    '--train-data-file', '-trdf', default='data/data-train/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--valid-data-file', '-vldf', default='data/data-valid/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--test-data-file', '-tsdf', default='data/data-test/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--sim-score-file', '-ssf', default='data/sim_resnik_yeast.txt',
    help='Semantic similarity scores for protein pairs (SemanticSimilarity.groovy)')
def main(go_file, train_data_file, valid_data_file, test_data_file, sim_score_file):
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
    train_data = load_data(train_data_file, prots_dict)
    valid_data = load_data(valid_data_file, prots_dict)
    trlabels = np.ones((len(proteins), len(proteins)), dtype=np.int32)
    for c, d in train_data:
        trlabels[c, d] = 0
    for c, d in valid_data:
        trlabels[c, d] = 0

    test_data = load_data(test_data_file, prots_dict)
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    n = len(test_data)
    labels = np.zeros((len(proteins), len(proteins)), dtype=np.int32) 
    ranks = {}
    franks = {}
    with ck.progressbar(test_data) as prog_data:
        for c, d in prog_data:
            labels[c, d] = 1
            index = rankdata(-sim[c, :], method='average')
            rank = index[d]
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            fil = sim[c, :] * (labels[c, :] | trlabels[c, :])
            index = rankdata(-fil, method='average')
            rank = index[d]
            if rank <= 10:
                ftop10 += 1
            if rank <= 100:
                ftop100 += 1
            fmean_rank += rank
            if rank not in franks:
                franks[rank] = 0
            franks[rank] += 1

        print()
        top10 /= n
        top100 /= n
        mean_rank /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n

        rank_auc = compute_rank_roc(ranks, len(proteins))
        frank_auc = compute_rank_roc(franks, len(proteins))
        print(f'{top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
        print(f'{ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc

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

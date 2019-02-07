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
    '--cls-embeds-file', '-cef', default='data/data-train/yeast_cls_embeddings.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/data-train/yeast_rel_embeddings.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=0.01,
    help='Loss margin')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
def main(go_file, data_file, cls_embeds_file, rel_embeds_file, margin,
         params_array_index):
    embedding_size = 100
    reg_norm = 1
    org = 'yeast'
    go = Ontology(go_file, with_rels=False)
    pai = params_array_index
    if params_array_index != -1:
        orgs = ['human', 'yeast']
        sizes = [50, 100, 200]
        margins = [-0.1, -0.01, 0.0, 0.01, 0.1]
        reg_norms = [1, 2]
        reg_norm = reg_norms[params_array_index % 2]
        params_array_index //= 2
        margin = margins[params_array_index % 5]
        params_array_index //= 5
        embedding_size = sizes[params_array_index % 3]
        params_array_index //= 3
        org = orgs[params_array_index % 2]
        print('Params:', org, embedding_size, margin, reg_norm)
        if org == 'human':
            pai = 0
            data_file = f'data/data-test/9606.protein.actions.v11.txt'
        else:
            pai = 1
        cls_embeds_file = f'data/noneg_{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_cls.pkl'
        rel_embeds_file = f'data/noneg_{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_rel.pkl'

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
    proteins = {}
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            proteins[k] = v
    rs = np.abs(embeds[:, -1]).reshape(-1, 1)
    embeds = embeds[:, :-1]

    prot_index = list(proteins.values())
    prot_rs = rs[prot_index, :]
    prot_embeds = embeds[prot_index, :]
    prot_dict = {v: k for k, v in enumerate(prot_index)}
    
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
    labels = {}
    preds = {}
    with ck.progressbar(data) as prog_data:
        for c, r, d in prog_data:
            c, r, d = prot_dict[classes[c]], relations[r], prot_dict[classes[d]]
            if r not in labels:
                labels[r] = np.zeros((len(prot_embeds), len(prot_embeds)), dtype=np.int32)
            if r not in preds:
                preds[r] = np.zeros((len(prot_embeds), len(prot_embeds)), dtype=np.float32)
            labels[r][c, d] = 1
            ec = prot_embeds[c, :]
            rc = prot_rs[c, :]
            er = rembeds[r, :]
            ec += er

            dst = np.linalg.norm(prot_embeds - ec.reshape(1, -1), axis=1)
            dst = dst.reshape(-1, 1)
            if rc > 0:
                overlap = np.maximum(0, (2 * rc - np.maximum(dst + rc - prot_rs - margin, 0)) / (2 * rc))
            else:
                overlap = (np.maximum(dst - prot_rs - margin, 0) == 0).astype('float32')
            
            edst = np.maximum(0, dst - rc - prot_rs - margin)
            res = (overlap + 1 / np.exp(edst)) / 2
            res = res.flatten()
            preds[r][c, :] = res
            index = rankdata(-res, method='average')
            rank = index[d]
            if rank == 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank /= n
    gl = np.zeros((len(prot_embeds), len(prot_embeds)), dtype=np.int32)
    gp = np.zeros((len(prot_embeds), len(prot_embeds)), dtype=np.int32)
    for i, l in labels.items():
        p = preds[i]
        gl = np.maximum(gl, l)
        gp = np.maximum(gp, p)
        # roc_auc = compute_roc(l, p)
        # fmax = compute_fmax(l, p)
        # print(rel_df['relations'][i], roc_auc)
        # print(fmax)
    print()
    roc_auc = compute_roc(gl, gp)
    print(f'{org} {embedding_size} {margin} {reg_norm} {top10:.2f} {top100:.2f} {mean_rank:.2f} {roc_auc:.2f}')
    
    
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

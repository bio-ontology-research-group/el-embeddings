#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
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
    '--data-file', '-df', default='go-normalized.txt',
    help='Data file')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/cls_embeddings.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/rel_embeddings.pkl',
    help='Relation embedings file')
def main(go_file, data_file, cls_embeds_file, rel_embeds_file):
    go = Ontology(go_file, with_rels=False)

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    embeds_list = cls_df['embeddings'].values
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    rs = np.abs(embeds[:, -1])
    embeds = embeds[:, :-1]

    data, classes, relations = load_data(data_file)

    classes = {v: k for k, v in classes.items()}
    relations = {v: k for k, v in relations.items()}

    n = len(data['nf1'])
    
    s = 0
    for c, d in data['nf1']:
        ec = embeds[c, :]
        rc = rs[c]
        ed = embeds[d, :]
        rd = rs[d]

        dst = np.linalg.norm(ec - ed)
        if dst + rc <= rd:
            s += 1
    print('Normal form 1', n, s, s / n)
    plot_embeddings(embeds, rs, classes)
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
        dst = dst + rs
        subs = (dst <= rc).astype(np.int8)
        preds[i, :] = subs

    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f = 2 * precision * recall / ( precision + recall)
    print(f, precision, recall)


def plot_embeddings(embeds, rs, classes):
    if embeds.shape[1] > 2:
        embeds = TSNE().fit_transform(embeds)
    go = Ontology('data/go.obo')
    bp = go.get_term_set(FUNC_DICT['bp'])
    mf = go.get_term_set(FUNC_DICT['mf'])
    cc = go.get_term_set(FUNC_DICT['cc'])
    
    fig, ax =  plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    for i in range(embeds.shape[0]):
        x, y = embeds[i, 0], embeds[i, 1]
        r = rs[i]
        ax.add_artist(plt.Circle(
            (x, y), r, color='blue', alpha=0.05,
            edgecolor='black'))
        ax.annotate(classes[i], xy=(x, y), fontsize=4, ha="center")
    ax.legend()
    ax.grid(True)
    plt.savefig('embeds.pdf')
    plt.show()

    
if __name__ == '__main__':
    main()

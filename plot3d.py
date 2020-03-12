#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

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
    '--cls-embeds-file', '-cef', default='data/cls_embeddings.pkl_test.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/rel_embeddings.pkl_test.pkl',
    help='Relation embedings file')
@ck.option(
    '--epoch', '-e', default='',
    help='Epoch embeddings')
def main(go_file, cls_embeds_file, rel_embeds_file, epoch):

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    print(nb_classes)
    nb_relations = len(rel_df)
    embeds_list = cls_df['embeddings'].values
    classes = {k: v for k, v in enumerate(cls_df['classes'])}
    rembeds_list = rel_df['embeddings'].values
    relations = {k: v for k, v in enumerate(rel_df['relations'])}
    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    rs = np.abs(embeds[:, -1])
    prot_rs = []
    for i, c in classes.items():
        if not c.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_rs.append(i)
    # print(len(prot_rs))
    # n, bins, patches = plt.hist(rs[prot_rs].flatten(), 100, facecolor='g', alpha=0.75)
    # plt.xlabel('Radiuses')
    # plt.ylabel('Length')
    # plt.title('Histogram of Radiuses')
    # plt.grid(True)
    # plt.show()
    # return

    embeds = embeds[:, :-1]

    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb

    plot_embeddings(embeds, rs, classes, epoch)

def plot_embeddings(embeds, rs, classes, epoch):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if embeds.shape[1] > 2:
        embeds = TSNE().fit_transform(embeds)
    
    fig =  plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    
    
    for i in range(embeds.shape[0]):
        if classes[i].startswith('owl:'):
            continue
        if classes[i] in {'<Maxat>', '<Aigerim>'}:
            continue
        a, b = embeds[i, 0], embeds[i, 1]
        r = rs[i]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v)) + a
        y = r * np.outer(np.sin(u), np.sin(v)) + b
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=colors[(i + 2) % len(colors)], rstride=4, cstride=4, linewidth=0, alpha=0.3)
        # ax.annotate(classes[i][1:-1], xy=(x, y + r + 0.03), fontsize=10, ha="center", color=colors[i % len(colors)])
    filename = 'embeds3d.pdf'
    plt.savefig(filename)
    plt.show()

    
if __name__ == '__main__':
    main()

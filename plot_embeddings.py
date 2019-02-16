#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque

from utils import Ontology, FUNC_DICT

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/cls_embeddings.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/rel_embeddings.pkl',
    help='Relation embedings file')
@ck.option(
    '--epoch', '-e', default='',
    help='Epoch embeddings')
def main(go_file, cls_embeds_file, rel_embeds_file, epoch):

    cls_df = pd.read_pickle(cls_embeds_file)
    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
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
    embeds = embeds[:, :-1]

    rsize = len(rembeds_list[0])
    rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
    for i, emb in enumerate(rembeds_list):
        rembeds[i, :] = emb

    plot_embeddings(embeds, rs, classes, epoch)

def plot_embeddings(embeds, rs, classes, epoch):
    if embeds.shape[1] > 2:
        embeds = TSNE().fit_transform(embeds)
    
    fig, ax =  plt.subplots()
    
    ax.set_xlim(-4, 2)
    ax.set_ylim(-3, 4)
    colors = plt.cm.get_cmap('hsv', embeds.shape[0])
    for i in range(embeds.shape[0]):
        if classes[i] == 'owl:Nothing':
            continue
        x, y = embeds[i, 0], embeds[i, 1]
        r = rs[i]
        ax.add_artist(plt.Circle(
            (x, y), r, fill=False, edgecolor=colors(i)))
        ax.annotate(classes[i], xy=(x, y), fontsize=6, ha="center")
    ax.legend()
    ax.grid(True)
    filename = 'embeds.pdf'
    if epoch:
        filename = f'embeds_{int(epoch):02d}.png' 
    plt.savefig(filename)
    # plt.show()

    
if __name__ == '__main__':
    main()

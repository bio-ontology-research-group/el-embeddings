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
    '--out-file', '-of', default='data/go-negatives.txt',
    help='Result file with negative GO relations')
def main(go_file, out_file):
    go = Ontology(go_file, with_rels=False)

    cc = get_top_classes(go, FUNC_DICT['cc'])    
    mf = get_top_classes(go, FUNC_DICT['mf'])
    bp = get_top_classes(go, FUNC_DICT['bp'])

    cc = list(map(lambda x: f'<http://purl.obolibrary.org/obo/{x.replace(":","_")}>', cc))
    mf = list(map(lambda x: f'<http://purl.obolibrary.org/obo/{x.replace(":","_")}>', mf))
    bp = list(map(lambda x: f'<http://purl.obolibrary.org/obo/{x.replace(":","_")}>', bp))
    f = open(out_file, 'w')
    for id1 in cc:
        for id2 in mf:
            for id3 in bp:
                f.write(id1 + '\t' + id2 + '\n')
                f.write(id2 + '\t' + id1 + '\n')
                f.write(id2 + '\t' + id3 + '\n')
                f.write(id3 + '\t' + id2 + '\n')
                f.write(id1 + '\t' + id3 + '\n')
                f.write(id3 + '\t' + id1 + '\n')
    f.close()            


def get_top_classes(go, go_id):
    res = []
    res.append(go_id)
    res += list(go.get(go_id)['children'])
    return res
    
if __name__ == '__main__':
    main()

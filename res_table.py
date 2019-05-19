#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque

logging.basicConfig(level=logging.INFO)

@ck.command()
def main():
    print('\\hline')
    print('Embedding Size & Margin & Raw Hits@10 & Filtered Hits@10 & Raw Hits@100 & Filtered Hits@100 & Raw Mean Rank & Filtered Mean Rank & Raw AUC & Filtered AUC\\\\')
    print('\\hline')
    for i in range(20, 40):
        with open(f'{i}.res') as f:
            lines = f.readlines()
            r1 = lines[-2].split()
            r2 = lines[-1].split()
            top10 = r1[4]
            top100 = r1[5]
            ftop10 = r2[4]
            ftop100 = r2[5]
            mr = float(r1[6])
            fmr = float(r2[6])
            auc = r1[7]
            fauc = r2[7]
            print(f'{r1[1]} & {r1[2]} & {top10} & {ftop10} & {top100} & {ftop100} & {mr:.0f} & {fmr:.0f} & {auc} & {fauc} \\\\')
            print('\\hline')
    
if __name__ == '__main__':
    main()

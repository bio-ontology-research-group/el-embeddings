#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import math
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import (
    Input,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='go-normalized.txt',
    help='Normalized ontology file (Normalizer.groovy)')
@ck.option(
    '--neg-data-file', '-ndf', default='data/go-negatives.txt',
    help='Negative subclass relations (generate_negatives.py)')
@ck.option(
    '--out-classes-file', '-ocf', default='data/cls_embeddings.pkl',
    help='Pandas pkl file with class embeddings')
@ck.option(
    '--out-relations-file', '-orf', default='data/rel_embeddings.pkl',
    help='Pandas pkl file with relation embeddings')
@ck.option(
    '--batch-size', '-bs', default=256,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1024,
    help='Training epochs')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='GPU Device ID')
@ck.option(
    '--embedding-size', '-es', default=100,
    help='Embeddings size')
@ck.option(
    '--reg-norm', '-rn', default=1,
    help='Regularization norm')
@ck.option(
    '--margin', '-m', default=0.01,
    help='Loss margin')
@ck.option(
    '--learning-rate', '-lr', default=0.01,
    help='Learning rate')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
@ck.option(
    '--loss-history-file', '-lhf', default='data/loss_history.csv',
    help='Pandas pkl file with loss history')
def main(data_file, neg_data_file, out_classes_file, out_relations_file,
         batch_size, epochs, device, embedding_size, reg_norm, margin,
         learning_rate, params_array_index, loss_history_file):
    # SLURM JOB ARRAY INDEX
    pai = params_array_index
    if params_array_index != -1:
        orgs = ['human', 'yeast']
        sizes = [50, 100, 200]
        margins = [-0.1, -0.01, 0.0, 0.01, 0.1]
        reg_norms = [1,]
        reg_norm = reg_norms[0]
        margin = margins[params_array_index % 5]
        params_array_index //= 5
        embedding_size = sizes[params_array_index % 3]
        params_array_index //= 3
        org = orgs[params_array_index % 2]
        print('Params:', org, embedding_size, margin, reg_norm)
        
        data_file = f'data/data-train/{org}-classes-normalized.owl'
        out_classes_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_cls.pkl'
        out_relations_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_rel.pkl'
        loss_history_file = f'data/{org}_{pai}_{embedding_size}_{margin}_{reg_norm}_loss.csv'
        
    train_data, valid_data, classes, relations = load_data(data_file, neg_data_file)
    nb_classes = len(classes)
    nb_relations = len(relations)
    nb_train_data = 0
    for key, val in train_data.items():
        nb_train_data = max(len(val), nb_train_data)
    train_steps = int(math.ceil(nb_train_data / (1.0 * batch_size)))
    train_generator = Generator(train_data, batch_size, steps=train_steps)

    nb_valid_data = 0
    for key, val in valid_data.items():
        nb_valid_data = max(len(val), nb_valid_data)
    valid_steps = int(math.ceil(nb_valid_data / (1.0 * batch_size)))
    valid_generator = Generator(valid_data, batch_size, steps=valid_steps)
    
    cls_dict = {v: k for k, v in classes.items()}
    rel_dict = {v: k for k, v in relations.items()}

    cls_list = []
    rel_list = []
    for i in range(nb_classes):
        cls_list.append(cls_dict[i])
    for i in range(nb_relations):
        rel_list.append(rel_dict[i])

    with tf.device('/' + device):
        nf1 = Input(shape=(2,), dtype=np.int32)
        nf2 = Input(shape=(3,), dtype=np.int32)
        nf3 = Input(shape=(3,), dtype=np.int32)
        nf4 = Input(shape=(3,), dtype=np.int32)
        dis = Input(shape=(3,), dtype=np.int32)
        neg = Input(shape=(2,), dtype=np.int32)
        el_model = ELModel(nb_classes, nb_relations, embedding_size, batch_size, margin, reg_norm)
        out = el_model([nf1, nf2, nf3, nf4, dis])
        model = tf.keras.Model(inputs=[nf1, nf2, nf3, nf4, dis], outputs=out)
        model.compile(optimizer='sgd', loss='mse')

        checkpointer = MyModelCheckpoint(
            out_classes_file=out_classes_file,
            out_relations_file=out_relations_file,
            cls_list=cls_list,
            rel_list=rel_list,
            monitor='val_loss')
        earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        logger = CSVLogger(loss_history_file)
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            workers=12,
            callbacks=[logger, earlystopper, checkpointer])


class ELModel(tf.keras.Model):

    def __init__(self, nb_classes, nb_relations, embedding_size, batch_size, margin=0.01, reg_norm=1):
        super(ELModel, self).__init__()
        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.margin = margin
        self.reg_norm = reg_norm
        self.batch_size = batch_size
        
        self.cls_embeddings = tf.keras.layers.Embedding(
            nb_classes,
            embedding_size + 1,
            input_length=1)
        self.rel_embeddings = tf.keras.layers.Embedding(
            nb_relations,
            embedding_size,
            input_length=1)
            
    def call(self, input):
        """Run the model."""
        nf1, nf2, nf3, nf4, dis = input
        loss1 = self.nf1_loss(nf1)
        loss2 = self.nf2_loss(nf2)
        loss3 = self.nf3_loss(nf3)
        loss4 = self.nf4_loss(nf4)
        loss_dis = self.dis_loss(dis)
        # loss_neg = self.neg_loss(neg)
        loss = loss1 + loss2 + loss3 + loss4 + loss_dis # + loss_neg
        return loss
   
    def loss(self, c, d):
        rc = tf.math.abs(c[:, -1])
        rd = tf.math.abs(d[:, -1])
        c = c[:, 0:-1]
        d = d[:, 0:-1]
        euc = tf.norm(c - d, axis=1)
        dst = tf.reshape(tf.nn.relu(euc + rc - rd - self.margin), [-1, 1])
        return dst + self.reg(c) + self.reg(d)

    def reg(self, x):
        res = tf.abs(tf.norm(x, axis=1) - self.reg_norm)
        res = tf.reshape(res, [-1, 1])
        return res
        
    def nf1_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        return self.loss(c, d)
    
    def nf2_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        e = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        e = self.cls_embeddings(e)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        re = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = e[:, 0:-1]
        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        dst2 = tf.reshape(tf.norm(x3 - x1, axis=1), [-1, 1])
        dst3 = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        rdst = tf.nn.relu(tf.math.minimum(rc, rd) - re)
        dst_loss = (tf.nn.relu(dst - sr)
                    + tf.nn.relu(dst2 - rc)
                    + tf.nn.relu(dst3 - rd)
                    + rdst - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3)

    def nf3_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        rd = tf.concat([r, tf.zeros((self.batch_size, 1), dtype=tf.float32)], 1)
        c = c + rd
        return self.loss(c, d) + self.reg(r)

    def nf4_loss(self, input):
        # R some C subClassOf D
        r = input[:, 0]
        c = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        rr = tf.concat([r, tf.zeros((self.batch_size, 1), dtype=tf.float32)], 1)
        c = c - rr
        # c - r should intersect with d
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        dst_loss = tf.nn.relu(dst - sr - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(r)
    

    def dis_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        return tf.nn.relu(sr - dst + self.margin) + self.reg(x1) + self.reg(x2)

    # def neg_loss(self, input, margin, reg_norm):
    #     c = input[:, 0]
    #     d = input[:, 1]
    #     c = self.cls_embeddings(c)
    #     d = self.cls_embeddings(d)
    #     rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
    #     rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
    #     x1 = c[:, 0:-1]
    #     x2 = d[:, 0:-1]
    #     x = x2 - x1
    #     dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
    #     return tf.nn.relu(rd - rc - dst + self.margin) + self.reg(x1) + self.reg(x2)
    

class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super(ModelCheckpoint, self).__init__()
        self.out_classes_file = kwargs.pop('out_classes_file')
        self.out_relations_file = kwargs.pop('out_relations_file')
        self.monitor = kwargs.pop('monitor')
        self.cls_list = kwargs.pop('cls_list')
        self.rel_list = kwargs.pop('rel_list')
        self.best = 1000
   
    def on_epoch_end(self, epoch, logs=None):
        # Save embeddings every 10 epochs
        current = logs.get(self.monitor)
        if math.isnan(current):
            return
        if current < self.best:
            self.best = current
            print(f'\n Saving embeddings {epoch + 1} {current}\n')
            el_model = self.model.layers[-1]
            cls_embeddings = el_model.cls_embeddings.get_weights()[0]
            rel_embeddings = el_model.rel_embeddings.get_weights()[0]

            cls_file = self.out_classes_file
            rel_file = self.out_relations_file
            # Save embeddings of every thousand epochs
            if (epoch + 1) % 1000 == 0:
                cls_file = f'{cls_file}_{epoch + 1}.pkl'
                rel_file = f'{rel_file}_{epoch + 1}.pkl'

            df = pd.DataFrame(
                {'classes': self.cls_list, 'embeddings': list(cls_embeddings)})
            df.to_pickle(cls_file)
                
            df = pd.DataFrame(
                {'relations': self.rel_list, 'embeddings': list(rel_embeddings)})
            df.to_pickle(rel_file)
            

        

class Generator(object):

    def __init__(self, data, batch_size=128, steps=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = steps
        self.start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.steps:
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf2_index = np.random.choice(
                self.data['nf2'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
            dis_index = np.random.choice(
                self.data['disjoint'].shape[0], self.batch_size)
            # neg_index = np.random.choice(
            #     self.data['negatives'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf2 = self.data['nf2'][nf2_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            dis = self.data['disjoint'][dis_index]
            # neg = self.data['negatives'][neg_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return ([nf1, nf2, nf3, nf4, dis], labels)
        else:
            self.reset()


def load_data(filename, filename_neg, index=True):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': []}
    with open(filename) as f:
        for line in f:
            # Ignore SubObjectPropertyOf
            if line.startswith('SubObjectPropertyOf'):
                continue
            # Ignore SubClassOf()
            line = line.strip()[11:-1]
            if not line:
                continue
            if line.startswith('ObjectIntersectionOf('):
                # C and D SubClassOf E
                it = line.split(' ')
                c = it[0][21:]
                d = it[1][:-1]
                e = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if e not in classes:
                    classes[e] = len(classes)
                form = 'nf2'
                if e == 'owl:Nothing':
                    form = 'disjoint'
                if index:
                    data[form].append((classes[c], classes[d], classes[e]))
                else:
                    data[form].append((c, d, e))
                
            elif line.startswith('ObjectSomeValuesFrom('):
                # R some C SubClassOf D
                it = line.split(' ')
                r = it[0][21:]
                c = it[1][:-1]
                d = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                if index:
                    data['nf4'].append((relations[r], classes[c], classes[d]))
                else:
                    data['nf4'].append((r, c, d))
            elif line.find('ObjectSomeValuesFrom') != -1:
                # C SubClassOf R some D
                it = line.split(' ')
                c = it[0]
                r = it[1][21:]
                d = it[2][:-1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                if index:
                    data['nf3'].append((classes[c], relations[r], classes[d]))
                else:
                    data['nf3'].append((c, r, d))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if index:
                    data['nf1'].append((classes[c], classes[d]))
                else:
                    data['nf1'].append((c, d))

    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])

    data['negatives'] = []
    with open(filename_neg, 'r') as f:
        for line in f:
            it = line.strip().split()
            c = it[0]
            d = it[1]
            if c not in classes:
                classes[c] = len(classes)
            if d not in classes:
                classes[d] = len(classes)
            if index:
                data['negatives'].append((classes[c], classes[d]))
            else:
                data['negatives'].append((c, d))
    data['negatives'] = np.array(data['negatives'])

    train_data = {}
    valid_data = {}
    split = 0.9
    for key, val in data.items():
        n = val.shape[0]
        train_n = int(split * n)
        index = np.arange(n)
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_data[key] = val[index[:train_n]]
        valid_data[key] = val[index[train_n:]]
    return train_data, valid_data, classes, relations

if __name__ == '__main__':
    main()

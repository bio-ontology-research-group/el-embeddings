#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import math
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

tf.enable_eager_execution()

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
def main(data_file, neg_data_file, out_classes_file, out_relations_file,
         batch_size, epochs, device, embedding_size, reg_norm, margin,
         learning_rate, params_array_index):
    # SLURM JOB ARRAY INDEX
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
        
        data_file = f'data/data-train/{org}-classes-normalized.owl'
        out_classes_file = f'data/noneg_{org}_{params_array_index}_{embedding_size}_{margin}_{reg_norm}_cls.pkl'
        out_relations_file = f'data/noneg_{org}_{params_array_index}_{embedding_size}_{margin}_{reg_norm}_rel.pkl'
        
    data, classes, relations = load_data(data_file, neg_data_file)
    nb_classes = len(classes)
    nb_relations = len(relations)
    nb_data = 0
    for key, val in data.items():
        nb_data = max(len(val), nb_data)
    steps = int(math.ceil(nb_data / (1.0 * batch_size)))
    generator = Generator(data, steps=steps)

    cls_dict = {v: k for k, v in classes.items()}
    rel_dict = {v: k for k, v in relations.items()}
    
    with tf.device('/' + device):
        model = ELModel(nb_classes, nb_relations, embedding_size, margin, reg_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        loss_history = []
        best_loss = 1000.0
        for epoch in range(epochs):
            loss = 0.0
            for batch, batch_data in enumerate(generator):
                input, labels = batch_data
                with tf.GradientTape() as tape:
                    logits = model(input)
                    loss_value = tf.losses.mean_squared_error(labels, logits)
                    loss += loss_value.numpy()
                print(f'Batch loss {loss_value.numpy()}', end='\r', flush=True)    
                loss_history.append(loss_value.numpy())
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables),
                    global_step=tf.train.get_or_create_global_step())
            if math.isnan(loss):
                print('NaN loss, exiting')
                break
            loss /= steps
            print(f'Epoch {epoch}: {loss}')

            # Save embeddings every 10 epochs and at the end
            if (epoch % 10 == 0 or epoch == epochs - 1) and best_loss > loss:
                logging.info(f'Loss improved from {best_loss} to {loss}')
                logging.info(f'Saving embeddings')
                cls_embeddings = model.cls_embeddings(
                    tf.range(nb_classes)).numpy()
                rel_embeddings = model.rel_embeddings(
                    tf.range(nb_relations)).numpy()
                cls_list = []
                rel_list = []
                for i in range(nb_classes):
                    cls_list.append(cls_dict[i])
                for i in range(nb_relations):
                    rel_list.append(rel_dict[i])

                df = pd.DataFrame(
                    {'classes': cls_list, 'embeddings': list(cls_embeddings)})
                df.to_pickle(out_classes_file)

                df = pd.DataFrame(
                    {'relations': rel_list, 'embeddings': list(rel_embeddings)})
                df.to_pickle(out_relations_file)


class ELModel(tf.keras.Model):

    def __init__(self, nb_classes, nb_relations, embedding_size, margin=0.01, reg_norm=1):
        super(ELModel, self).__init__()
        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.margin = margin
        self.reg_norm = reg_norm
        
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
        nf1, nf2, nf3, nf4, dis, neg = input
        loss1 = self.nf1_loss(nf1, self.margin, self.reg_norm)
        loss2 = self.nf2_loss(nf2, self.margin, self.reg_norm)
        loss3 = self.nf3_loss(nf3, self.margin, self.reg_norm)
        loss4 = self.nf4_loss(nf4, self.margin, self.reg_norm)
        loss_dis = self.dis_loss(dis, self.margin, self.reg_norm)
        # loss_neg = self.neg_loss(dis, self.margin, self.reg_norm)
        loss = loss1 + loss2 + loss3 + loss4 + loss_dis # + loss_neg
        return loss
   
    def loss(self, c, d, margin, reg_norm):
        rc = tf.math.abs(c[:, -1])
        rd = tf.math.abs(d[:, -1])
        c = c[:, 0:-1]
        d = d[:, 0:-1]
        euc = tf.norm(c - d, axis=1)
        dst = tf.reshape(tf.nn.relu(euc + rc - rd - margin), [-1, 1])
        # Regularization
        reg = tf.abs(tf.norm(c, axis=1) - reg_norm) + tf.abs(tf.norm(d, axis=1) - reg_norm)
        reg = tf.reshape(reg, [-1, 1])
        return dst + reg
    
    def nf1_loss(self, input, margin, reg_norm):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        return self.loss(c, d, margin, reg_norm)
    
    def nf2_loss(self, input, margin, reg_norm):
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
                + rdst)
        reg = (tf.abs(tf.norm(x1, axis=1) - reg_norm)
               + tf.abs(tf.norm(x2, axis=1) - reg_norm)
               + tf.abs(tf.norm(x3, axis=1) - reg_norm))
        reg = tf.reshape(reg, [-1, 1])
        return dst_loss + reg
        
                
    def nf3_loss(self, input, margin, reg_norm):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        r = tf.concat([r, tf.zeros((r.shape[0], 1), dtype=tf.float32)], 1)
        c = c + r
        return self.loss(c, d, margin, reg_norm)

    def nf4_loss(self, input, margin, reg_norm):
        # R some C subClassOf D
        r = input[:, 0]
        c = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        r = tf.concat([r, tf.zeros((r.shape[0], 1), dtype=tf.float32)], 1)
        c = c - r
        # c - r should intersect with d
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        dst_loss = tf.nn.relu(dst - sr - margin)
        reg = tf.abs(tf.norm(x1, axis=1) - reg_norm) + tf.abs(tf.norm(x2, axis=1) - reg_norm)
        reg = tf.reshape(reg, [-1, 1])
        return dst_loss + reg
    

    def dis_loss(self, input, margin, reg_norm):
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
        reg = tf.abs(tf.norm(x1, axis=1) - reg_norm) + tf.abs(tf.norm(x2, axis=1) - reg_norm)
        reg = tf.reshape(reg, [-1, 1])
        return tf.nn.relu(sr - dst + margin) + reg

    def neg_loss(self, input, margin, reg_norm):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        reg = tf.abs(tf.norm(x1, axis=1) - reg_norm) + tf.abs(tf.norm(x2, axis=1) - reg_norm)
        reg = tf.reshape(reg, [-1, 1])
        return tf.nn.relu(rd - rc - dst + margin) + reg
    
        

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
            neg_index = np.random.choice(
                self.data['negatives'].shape[0], self.batch_size)
            nf1 = tf.convert_to_tensor(self.data['nf1'][nf1_index])
            nf2 = tf.convert_to_tensor(self.data['nf2'][nf2_index])
            nf3 = tf.convert_to_tensor(self.data['nf3'][nf3_index])
            nf4 = tf.convert_to_tensor(self.data['nf4'][nf4_index])
            dis = tf.convert_to_tensor(self.data['disjoint'][dis_index])
            neg = tf.convert_to_tensor(self.data['negatives'][neg_index])
            labels = tf.zeros((self.batch_size, 1), dtype=tf.float32)
            self.start += 1
            return ((nf1, nf2, nf3, nf4, dis, neg), labels)
        else:
            self.reset()
            raise StopIteration()


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
    
    return data, classes, relations

if __name__ == '__main__':
    main()

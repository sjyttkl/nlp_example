# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     model2.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/10/31 19:32
   Description :  
==================================================
"""

import tensorflow as tf
from crf_layer import CRF
import numpy as np
import os


class BiLstmCrfModel:
    def __init__(self, max_len, vocab_size, embedding_dim, lstm_units, class_nums, embedding_matrix=None):
        super(BiLstmCrfModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.class_nums = class_nums
        self.embedding_matrix = embedding_matrix  # 预训练词向量
        self.embedding_dim = embedding_dim
        if self.embedding_matrix is not None:
            self.vocab_size, self.embedding_dim = self.embedding_matrix.shape
        self._build()
    def _build(self):
        inputs = tf.keras.layers.Input(shape=(self.max_len,),name='input_ids', dtype=tf.int32)

        targets = tf.keras.layers.Input(shape=(self.max_len,), name='target_ids', dtype=tf.int32)
        # seq_lens = tf.reduce_sum(tf.cast(tf.cast(inputs, tf.bool), tf.int32),-1)

        seq_lens = tf.keras.layers.Input(shape=(), name='input_lens', dtype=tf.int32)  # 真实长度

        x = tf.keras.layers.Masking(mask_value=0)(inputs)  # 表示 为0 的数据，需要进行mask，在lstm中会跳过计算，先进行了padding（使用0），后面使用这个进行标记，
        x = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,trainable=True,weights=self.embedding_matrix,
                                      mask_zero=True)(inputs) #  , ,
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)  # 对lstm 每个字（每个token）对输出做droput

        logits = tf.keras.layers.Dense(self.class_nums, name='dense')(x)
        loss = CRF(label_size=self.class_nums, name="crf_layer")(logits, targets,seq_lens,)

        model = tf.keras.Model(inputs=[inputs, targets, seq_lens], outputs=loss)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        # 自定义Loss
        # model.compile(loss=CustomLoss(), optimizer='adam')
        # 或者使用lambda表达式
        model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=opt)
        print(model.summary())
        # Tensorboard, earlystopping, ModelCheckpoint
        logdir = './graph_def_and_weights'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        output_model_file = os.path.join(logdir,
                                         "bilstm_crf.h5")

        self.callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),
            tf.keras.callbacks.ModelCheckpoint(output_model_file,
                                               save_best_only=True,  # 表示只保存最好的模型，可能是中间的一个模型
                                               save_weights_only=True),
            # 这是save_weights_only表示只保存参数，不保存图结构，默认格式是参数和图结构都保存
            tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
        ]
        self.model = model

    def train(self,
              train_datas,
              train_label,
              train_seq_lens,
              train_labels,
              train_datas_valid,
              train_label_valid,
              train_seq_lens_valid,
              labels_valid,
              batch_size,
              epochs ):
        self.model.fit(x=[train_datas, train_label, train_seq_lens], y=train_labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       # validation_split=0.1,
                       validation_data=([train_datas_valid, train_label_valid, train_seq_lens_valid], labels_valid),
                       callbacks=self.callbacks
                     )
        #
        #

class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss, pred = y_pred
        return loss

# 自定义Loss
# model.compile(loss=CustomLoss(), optimizer='adam')
# 或者使用lambda表达式
# model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')

#
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 20)]              0
# _________________________________________________________________
# masking (Masking)            (None, 20)                0
# _________________________________________________________________
# embedding (Embedding)        (None, 20, 30)            132360
# _________________________________________________________________
# bidirectional (Bidirectional (None, 20, 128)           48640
# _________________________________________________________________
# time_distributed (TimeDistri (None, 20, 128)           0
# _________________________________________________________________
# tf.cast (TFOpLambda)         (None, 20, 128)           0
# _________________________________________________________________
# crf_layer (CRF)              (None, 20)                1872
# =================================================================
# Total params: 182,872
# Trainable params: 182,872
# Non-trainable params: 0
# _________________________________________________________________

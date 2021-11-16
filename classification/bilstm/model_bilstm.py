# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     model.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/25 20:52
   Description :  
==================================================
"""
import tensorflow as tf
import numpy as np
import os

class MyModel:
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size_char,
                 vocab_size_label,
                 use_crf):
        self.embedding_dim = embedding_dim
        self.vocab_size_char = vocab_size_char
        self.hidden_dim = hidden_dim
        self.vocab_size_label = vocab_size_label
        self._build()

    def _build(self):
        input_layer = tf.keras.Input(shape=(None,), dtype=tf.float32)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size_char,
            output_dim=self.embedding_dim, name="embedding_layer")

        input_embed = embedding_layer(input_layer)
        forward_layer = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=False)
        backward_layer = tf.keras.layers.LSTM(self.hidden_dim, activation='relu', return_sequences=False,
                                              go_backwards=True)
        bilstm_layer = tf.keras.layers.Bidirectional(layer=forward_layer,
                                                     backward_layer=backward_layer, merge_mode="sum",name='Bi-LSTM')
        input_bilstm = bilstm_layer(input_embed)

        prediction_layer = tf.keras.layers.Dense(self.vocab_size_label,activation=tf.nn.softmax,name="predictions")
        # sigmoid激活函数应用于多标签分类（一个样本可能属于两个 label)  将sigmoid激活函数应用于多标签分类时，其损失函数应设置为binary_crossentropy（交叉熵）； 一般不用来做多类分类，而是用来做二分类的
        # softmax激活函数应用于多分类(每个样本一个label）
        # 我们平时在给新闻贴标签的时候，就有可能把一篇文章分为经济和文化两个类别。因此，多标签问题在我们的日常生活中也是很常见的
        prediction = prediction_layer(input_bilstm)

        model = tf.keras.Model(input_layer,prediction)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])
        # tf.keras.losses.sparse_categorical_crossentropy
        print(model.summary())
        # Tensorboard, earlystopping, ModelCheckpoint
        logdir = './graph_def_and_weights'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        output_model_file = os.path.join(logdir,
                                         "fashion_mnist_weights.h5")
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(logdir),
            tf.keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,  # 表示只保存最好的模型，可能是中间的一个模型
                                            save_weights_only=True),
            # 这是save_weights_only表示只保存参数，不保存图结构，默认格式是参数和图结构都保存
            tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
        ]
        self.model = model

    def train(self,x_train,
                    y_train,
                    x_valid,
                    y_valid,
                    batch_size,
                    epochs,):
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_valid, y_valid),
                       callbacks = self.callbacks)

        # self.model.save(filepath="my_model.h5")
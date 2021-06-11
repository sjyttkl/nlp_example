# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     model_bilstm_attention.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/30 00:27
   Description :
   """
# 由于Keras目前还没有现成的Attention层可以直接使用，我们需要自己来构建一个新的层函数。
# Keras自定义的函数主要分为四个部分，分别是：
# init：初始化一些需要的参数
# bulid：具体来定义权重是怎么样的
# call：核心部分，定义向量是如何进行运算的
# compute_output_shape：定义该层输出的大小
# 推荐文章：
#     https://blog.csdn.net/huanghaocs/article/details/95752379
#     https://zhuanlan.zhihu.com/p/29201491
"""
==================================================
"""

import tensorflow as tf
import numpy as np
import os

# tf.keras.layers.Attention
from keras import initializers
from keras import constraints
from keras import activations
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
K.clear_session()

class AttentionLayer(tf.keras.layers):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1)) #[attention_size, 1]
        H = K.tanh(K.dot(inputs, self.W) + self.b) #[batch_size,time_step,attention_size]
        score = K.softmax(K.dot(H, self.V), axis=1) # [batch_size,time_step,1]
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    #这个函数是告诉我经过运算之后热输出的形状
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

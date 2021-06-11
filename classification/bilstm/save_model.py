# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     save_model.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/30 00:12
   Description :  
==================================================
"""
import tensorflow as tf
import  numpy as np
import os

from bilstm.model_bilstm import MyModel

model = MyModel(embedding_dim=300,
                hidden_dim=32,
                vocab_size_char=5776,
                vocab_size_label=10,
                use_crf=False)

logdir = './graph_def_and_weights'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "classifier_weights.h5")

model.model.load_weights(output_model_file)  #这里我们先定义好模型的结构，然后使用load_weights函数载入参数
tf.saved_model.save(model.model, "./keras_saved_graph")

#打印一下模型的签名
loaded_saved_model = tf.saved_model.load('./keras_saved_graph')
print(list(loaded_saved_model.signatures.keys()))

#打印模型 服务推断接口
inference = loaded_saved_model.signatures['serving_default']
print(inference)
print(inference.structured_outputs)

##输入测试集的第一个样本进行测试
results = inference(tf.constant(np.ones((10,200))))  #输入测试集的第一个样本进行测试
print(results['dense_2'])
#可以看出inference其实是一个函数句柄，他可以接收一个参数，获得一个输出，输出是字典dict，字典中的有一个key["dense_2"]
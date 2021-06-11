# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     AttentionLayer.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/6/10 21:53
   Description :  Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
由于Keras目前还没有现成的Attention层可以直接使用，我们需要自己来构建一个新的层函数。
# Keras自定义的函数主要分为四个部分，分别是：
# init：初始化一些需要的参数
# bulid：具体来定义权重是怎么样的
# call：核心部分，定义向量是如何进行运算的
# compute_output_shape：定义该层输出的大小
# 推荐文章：
#     https://blog.csdn.net/huanghaocs/article/details/95752379
#     https://zhuanlan.zhihu.com/p/29201491
https://www.cnblogs.com/yifanrensheng/category/1758378.html
https://blog.csdn.net/weixin_40849273/article/details/88576507
==================================================
"""
import tensorflow as tf
from keras import backend as K

class AttentionBaseLayer(tf.keras.layers.Layer):
    # 初始化attetion_size，即当前层输出元素的个数
    def __init__(self,attetion_size=None,**kwargs):
        self.attention_size = attetion_size
        super(AttentionBaseLayer, self).__init__(**kwargs)

    #这个需要覆盖下，不然tf，不能完整的保存模型，在加载模型的时候会出错
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention_size': self.attention_size
        })
        return config
    #build() 用来初始化定义weights, 这里可以用父类的self.add_weight() 函数来初始化数据, 该函数必须将 self.built 设置为True, 以保证该 Layer 已经成功 build , 通常如上所示, 使用 super(MyLayer, self).build(input_shape) 来完成
    def build(self,input_shape): #在build()中增删参数
        assert len(input_shape) == 3 #[batch_size,time_step,hidden_size]
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
        self.W = self.add_weight(name="att_weight",shape=(hidden_size,1),initializer='uniform',
                                 trainable=True)

        super(AttentionBaseLayer, self).build(input_shape)## 一定要在最后调用它
        #
        #self.built = True

    #在call()中更改计算方式、激活函数等等。
    def call(self, inputs, **kwargs):
        M = K.tanh(inputs)  #[batch_size,time_step,hidden_size]
        newM = tf.matmul(M,self.W) # [batch_size,time_step,1]    K.dot这个有问题。不建议使用
        alpha = K.softmax(newM,1)
        #[batch_size,hidden_size,time_step] * [batch_size,time_step,1]
        outputs = tf.matmul(tf.transpose(inputs, [0, 2, 1]),alpha) #[batch_size,hidden_size,1]  K.dot这个有问题。不建议使用
        outputs = tf.reshape(outputs,[-1, self.attention_size])
        outputs = tf.tanh(outputs)#[batch_size, hidden_size]

        return outputs
    #这个函数是告诉我经过运算之后热输出的形状
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


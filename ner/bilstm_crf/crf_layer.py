# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     crf_layer.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/8/1 00:27
   Description :
    https://blog.csdn.net/qq_41837900/article/details/100201109
   https://blog.csdn.net/qq_41837900/article/details/100201109
==================================================
"""

# tensorflow中的 有关crf 的函数：
#
# tf.contrib.crf.crf_sequence_score #计算标签序列的非规范化分数
# tf.contrib.crf.crf_log_norm #计算CRF的标准化.
# tf.contrib.crf.crf_log_likelihood #计算 CRF 中标签序列的对数似然.
# tf.contrib.crf.crf_unary_score #计算标签序列的一元分数.
# tf.contrib.crf.crf_binary_score #计算 CRF 中标签序列的二进制分数.
# tf.contrib.crf.CrfForwardRnnCell #计算线性链 CRF 中的 alpha 值.
# tf.contrib.crf.viterbi_decode #解码 TensorFlow 之外的标记的最高得分序列.这只能在测试时使用.

#
#
# 这里我们需要利用到的有关CRF的API其实只有两个，它们分别是：
# tfa.text.crf_log_likelihood()
#  tf.contrib.crf.viterbi_decode
#  它们分别用于计算标注序列的对数似然，以及根据给定参数解码标注序列。
# tfa.text.crf_decode()
import tensorflow as tf
import tensorflow_addons as tfa



class CRF(tf.keras.layers.Layer):

    def __init__(self, label_size, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.tags_num = label_size  # 表示 tags总数
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)), name="transition")

    # 这个需要覆盖下，不然tf，不能完整的保存模型，在加载模型的时候会出错
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tags_num': self.tags_num
        })
        return config

    # build() 用来初始化定义weights, 这里可以用父类的self.add_weight() 函数来初始化数据, 该函数必须将 self.built 设置为True, 以保证该 Layer 已经成功 build , 通常如上所示, 使用 super(MyLayer, self).build(input_shape) 来完成
    def build(self, input_shape, ):  # 在build()中增删参数
        # inputs = np.array([[[-3, 5, -1], [1, 1, 3], [-1, -2, 4], [0, 0, 0]]], dtype=np.float32)
        # assert len(input_shape[0]) == len(self.output_seq_len[0])
        # assert len(input_shape[0]) == len(self.input_shape[0])

        super(CRF, self).build(input_shape)  ## 一定要在最后调用它
        # self.built = True

    @tf.function  # 会把普通python计算变成 graph
    def call(self, inputs, labels, seq_lens):
        log_likelihood, self.trans_params = tfa.text.crf_log_likelihood(
            inputs, labels, seq_lens,
            transition_params=self.trans_params)
        loss = tf.reduce_sum(-log_likelihood)
        return loss

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:2]
        return output_shape + self.tags_num

    @property
    def _compute_dtype(self):
        # fixed output dtype from underline CRF functions
        return tf.int32

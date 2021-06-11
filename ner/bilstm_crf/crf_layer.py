# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     crf_layer.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/6/10 23:59
   Description :  CRF.
    https://blog.csdn.net/qq_41837900/article/details/100201109
   https://blog.csdn.net/qq_41837900/article/details/100201109
==================================================
 tensorflow中的 有关crf 的函数：
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
"""
import tensorflow as tf
import tensorflow_addons as tfa

class CRF(tf.keras.layers.Layer):
    def __init__(self, logits_seq, tag_indices, inputs_seq_len, **kwargs):
        super().__init__(**kwargs)
        self.output_seq = tag_indices
        self.output_seq_len = inputs_seq_len
        super(AttentionBaseLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention_size': self.attention_size
        })
        return config

    def build(self, input_shape):
        assert len(input_shape[0]) == len(self.output_seq_len[0])
        assert len(input_shape[0]) == len(self.input_shape[0])

        super(CRF, self).build(input_shape)## 一定要在最后调用它

    def call(self,inputs,train=True,**kwargs):

        transition_matrix  = None
        if(train):
            # log_likehihood A [batch_size] Tensor ,最大似然估计
            # transition_matrix A [num_tags, num_tags] transition matrix.
            log_likelihood, transition_matrix = tfa.text.crf.crf_log_likelihood(inputs=inputs,
                                                                                tag_indices=self.output_seq,
                                                                                sequence_lengths=self.output_seq_len)
            return log_likelihood
        else:
            # preds_seq ,预测出的序列
            # crf_scores ，crf分数
            preds_seq, crf_scores = tfa.text.crf.crf_decode(inputs, transition_matrix, self.inputs_seq_len)
            return preds_seq

    def compute_output_shape(self, input_shape):
        return input_shape[0]
# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     predict.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/11/16 10:10
   Description :  
==================================================
"""
from loader import char_mapping, tag_mapping


import tensorflow as tf
from model import BiLstmCrfModel
import tensorflow_addons as tfa
import numpy as np
import pickle

import os

def saveModel():
    model = BiLstmCrfModel(
        max_len=100,
        vocab_size=4412,  # 4412
        embedding_dim=300,
        lstm_units=64,
        class_nums=13,
        embedding_matrix=None
    )

    logdir = './graph_def_and_weights'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir,
                                     "bilstm_crf.h5")

    model.model.load_weights(output_model_file)  # 这里我们先定义好模型的结构，然后使用load_weights函数载入参数
    tf.saved_model.save(model.model, "./keras_saved_graph")

    # 打印一下模型的签名
    loaded_saved_model = tf.saved_model.load('./keras_saved_graph')
    print(list(loaded_saved_model.signatures.keys()))

    # 打印模型 服务推断接口
    inference = loaded_saved_model.signatures['serving_default']
    print(inference)
    print(inference.structured_outputs)


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    # 如果需要继续训练，需要下面的重新compile
    # model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')
    trans_params = model.get_layer('crf_layer').get_weights()[0]
    # print(trans_params)
    # 如果需要继续训练，需要下面的重新compile
    # model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')
    return model, trans_params


def predict(model, inputs, input_lens, trans_params):
    # 获得BiLSTM的输出logits
    sub_model = tf.keras.models.Model(inputs=model.get_layer('input_ids').input,
                                      outputs=model.get_layer('dense').output)
    logits = sub_model.predict(inputs)
    # 获取CRF层的转移矩阵
    # crf_decode：viterbi解码获得结果
    pred_seq, viterbi_score = tfa.text.crf_decode(logits, trans_params, input_lens)
    return pred_seq

def get_char_id(map_file):
    with open(map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    return char_to_id, id_to_char, tag_to_id, id_to_tag
# 对预测结果进行命名实体解析和提取
def get_valid_nertag(input_data, result_tags):
    result_words = []
    start, end = 0, 1  # 实体开始结束位置标识
    tag_label = "O"  # 实体类型标识
    for i, tag in enumerate(result_tags):
        if tag.startswith("B"):
            if tag_label != "O":  # 当前实体tag之前有其他实体
                result_words.append((input_data[start: end], tag_label))  # 获取实体
            tag_label = tag.split("-")[1]  # 获取当前实体类型
            start, end = i, i + 1  # 开始和结束位置变更
        elif tag.startswith("I"):
            temp_label = tag.split("-")[1]
            if temp_label == tag_label:  # 当前实体tag是之前实体的一部分
                end += 1  # 结束位置end扩展
        elif tag == "O":
            if tag_label != "O":  # 当前位置非实体 但是之前有实体
                result_words.append((input_data[start: end], tag_label))  # 获取实体
                tag_label = "O"  # 实体类型置"O"
            start, end = i, i + 1  # 开始和结束位置变更
    if tag_label != "O":  # 最后结尾还有实体
        result_words.append((input_data[start: end], tag_label))  # 获取结尾的实体
    return result_words


if __name__ == "__main__":
    saveModel()
    model, trans_params = load_model("./keras_saved_graph")
    maxlen = 100
    sentence = "去年十二月二十四日，市委书记张敬涛召集县市主要负责同志研究信访工作时，提出:"

    map_file = "maps.pkl"
    pre_emb = True
    vocab2idx, id_to_char, tag_to_id, idx2label = get_char_id(map_file)

    sent_chars = list(sentence)
    sent2id = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sent_chars]
    sent2id_new = np.array([ sent2id[:maxlen] + [0] * (maxlen - len(sent2id)) ])
    inputs_lens = np.array([np.sum(np.sign(sent2id_new))])


    pred_seq = predict(model, sent2id_new, inputs_lens,trans_params)
    print(pred_seq)

    y_label = pred_seq.numpy().reshape(1, -1)[0]
    print(y_label)
    y_ner = [idx2label[i] for i in y_label][:len(sent_chars)]

    print(sent2id)
    print(y_ner)

    result_words = get_valid_nertag(sent_chars, y_ner)
    for (word, tag) in result_words:
        print("".join(word), tag)
# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train_lstm.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/26 18:36
   Description :  
==================================================
"""

import logging
import os
import tensorflow as tf


from utils import DataProcessor_LSTM
from utils import load_vocabulary
# from model_bilstm import MyModel
from model_bilstm_attention import MyModel


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))


#set logging
log_file_path = "../ckpt/run_bilstm_attention.log"
if os.path.exists(log_file_path):os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()  #流handler 能够将日志信息输出到sys.stdout, sys.stderr 或者类文件对象
chlr.setFormatter(formatter)

fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)

logger.addHandler(chlr)
logger.addHandler(fhlr)


logger.info("loading vocab...")
path = "/Users/songdongdong/workSpace/datas/learning_data/中文文本分类/"
#path="/home/jovyan/data/中文文本分类/"
vocab_label = "vocab_char_label.txt"
vocab_char = "vocab_char.txt"
vocab_path_char = path + vocab_char
vocab_path_char_label = path + vocab_label
w2i_char, i2w_char = load_vocabulary(vocab_path_char)
w2i_label, i2w_label = load_vocabulary(vocab_path_char_label)

logger.info("loading data...")

#path = "/Users/songdongdong/workSpace/datas/learning_data/中文文本分类/"
train_file = "cnews.train.txt"
test_file = "cnews.test.txt"
val_file = "cnews.val.txt"
data_processor_train = DataProcessor_LSTM(
    path + train_file,
    path + train_file + "_bio",
    w2i_char,
    w2i_label,
    shuffling=True)

data_processor_valid = DataProcessor_LSTM(
    path + train_file,
    path + train_file + "_bio",#这个字段
    w2i_char,
    w2i_label,
    shuffling=True)


logger.info("building model ...")

print(len(w2i_char),len(w2i_label))
# print(w2i_char,w2i_label)
#5776,10
# print
model = MyModel(embedding_dim=300,
                hidden_dim=64,
                vocab_size_char=len(w2i_char),
                vocab_size_label=len(w2i_label),
                use_crf=False)

logger.info("start training...")


(inputs_seq_batch,
 inputs_seq_len_batch,
 outputs_seq_batch)= data_processor_train.get_batch(1000)
(inputs_seq_batch_valid,
 inputs_seq_len_batch_valid,
 outputs_seq_batch_valid)= data_processor_train.get_batch(1000)

print(inputs_seq_batch.shape,outputs_seq_batch.shape) #(1000, 12456) (1000, 12)
model.train(inputs_seq_batch,outputs_seq_batch,
            inputs_seq_batch_valid,
            outputs_seq_batch_valid,32,20)
#模型保存称pb
model.save("./keras_saved_graph")
# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     train.py.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/8/1 11:27
   Description :  
==================================================
"""
import tensorflow as tf
import os
import logging
import pickle
import itertools
from collections import OrderedDict

from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset

from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

from model import MyModel
from model2 import BiLstmCrfModel
import numpy as np
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))

# set logging
log_file_path = "../ckpt/run_bilstm_crf.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()  # 流handler 能够将日志信息输出到sys.stdout, sys.stderr 或者类文件对象
chlr.setFormatter(formatter)

fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)

logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info("loading vocab...")

path = "/Users/songdongdong/workSpace/datas/learning_data/ner/"
# path="/home/jovyan/data/ner/"

# vocab_label = "vocab_char_label.txt"
# vocab_char = "vocab_char.txt"
# vocab_path_char = path + vocab_char
# vocab_path_char_label = path + vocab_label
# w2i_char, i2w_char = load_vocabulary(vocab_path_char)
# w2i_label, i2w_label = load_vocabulary(vocab_path_char_label)
train_file = path + "example.train"
dev_file = path + "example.dev"
test_file = path + "example.test"
emb_file = path + "wiki_100.utf8"

tag_schema = "iobes"
map_file = "maps.pkl"
pre_emb = True
lower = True
batch_size = 64
# laod data
train_sentences = load_sentences(train_file, True,
                                 False)  # example.train [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O']]
dev_sentences = load_sentences(dev_file, True, False)
test_sentences = load_sentences(test_file, True, False)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_schema)  # iobes
update_tag_scheme(test_sentences, tag_schema)  # iobes

# create maps if not exist
if not os.path.isfile(map_file):
    # create dictionary for word
    if pre_emb:
        dico_chars_train = char_mapping(train_sentences, True)[0]
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(
            dico_chars_train.copy(),
            emb_file,
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in test_sentences])
            )
        )
    else:
        _c, char_to_id, id_to_char = char_mapping(train_sentences, lower)

    # Create a dictionary and a mapping for tags
    _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    with open(map_file, "wb") as f:
        pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
else:
    with open(map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

# prepare data, get a collection of list containing index
train_data = prepare_dataset(
    train_sentences, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, char_to_id, tag_to_id, lower
)
print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

max_len=100
train_manager = BatchManager(train_data, batch_size,max_len)
dev_manager = BatchManager(dev_data, batch_size,max_len)
test_manager = BatchManager(test_data, batch_size,max_len)


model = BiLstmCrfModel(
    max_len=max_len,
    vocab_size=len(char_to_id),  # 4412
    embedding_dim=300,
    lstm_units=64,
    class_nums=len(tag_to_id), #13
    embedding_matrix=None
)




train_X = train_manager.iter_batch(True)
dev_X = dev_manager.iter_batch(True)

train_x,train_x_len,train_y = train_manager.get_batch(train_X,batch_size)
dev_x,dev_x_len,dev_y = dev_manager.get_batch(dev_X,batch_size)

train_labels = np.ones(len(train_x)) # 这里只是随意设置个 ，因为
valid_labels = np.ones(len(dev_x)) # 这里只是随意设置个 ，因为



# 训练
model.train(train_x,
            train_y,
            train_x_len,
            train_labels,
            dev_x,
            dev_y,
            dev_x_len,
            valid_labels,
            batch_size,
            10)


# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     utils.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/25 21:56
   Description :  
==================================================
"""
import os
import random
import numpy as np
import re

#path = "/Users/songdongdong/workSpace/datas/learning_data/中文文本分类/"


def load_data(path, vocab_char, vocab_label):
    for file_name in os.listdir(path):
        if (file_name.endswith("train.txt")):
            train_text = open(path + file_name, "r", encoding="utf-8").readlines()
        elif (file_name.endswith("val.txt")):
            val_text = open(path + file_name, "r", encoding="utf-8").readlines()
        else:
            test_text = open(path + file_name, "r", encoding="utf-8").readlines()
    vacab_text = []
    vacab_text.extend(train_text)
    vacab_text.extend(val_text)
    vacab_text.extend(test_text)

    print("load vocab from: {}, containing words: {}".format(path, len(vacab_text)))
    labels = []
    vocab_text = []
    for line in vacab_text:
        if (len(line.strip()) < 5):
            continue
        label = line.split("\t")[0].strip()
        if (len(label) != 2):
            print(label)
            print(line)

        labels.append(label)
        line = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", line)
        # print(line)
        line_re = list(line.replace("", "").replace("\t", "").replace("\n", ""))
        # print(line_re)
        vocab_text.extend(line_re)
        # print(vocab_text)
    # print(len(vocab_text))
    vocab_text = set(vocab_text)
    labels = set(labels)
    print(len(vocab_text))
    # print(vocab_text)
    vocab_text = list(vocab_text)
    labels = list(labels)
    with open(path + vocab_char, "w", encoding="utf-8") as file:
        for word in vocab_text:
            file.write(word.strip() + "\n")
    with open(path + vocab_label, "w", encoding="utf-8") as file:
        for word in labels:
            file.write(word.strip() + "\n")


#########################
####### Vocabulary #######
##########################

def load_vocabulary(path):
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    if (path.find("label") >= 0):
        for i, w in enumerate(vocab):
            w2i[w] = i
            i2w[i] = w
    else:
        i2w[0] = "[UNK]"
        w2i["[UNK]"] = 0  # 未登录词
        i2w[1] = "[PAD]"  # 填充
        w2i["[PAD]"] = 1
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
    return w2i, i2w


class DataProcessor_LSTM(object):
    def __init__(self,
                 input_seq_path,
                 output_seq_path,
                 w2i_char,
                 w2i_label,
                 shuffling=False):

        inputs_seq = []
        outputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if (len(line.strip()) < 5):
                    continue
                text = line.split("\t")
                label = text[0]
                data = text[1]
                data = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", data)
                seq_data = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in list(data)]
                seq = [w2i_label[label]]
                inputs_seq.append(seq_data)
                outputs_seq.append(seq)

        # outputs_seq = []
        # with open(output_seq_path, "r", encoding="utf-8") as f:
        #     for line in f.readlines():
        #         if (len(line.strip()) < 5):
        #             continue
        #         text = line.split("\t")
        #         label = text[0]
        #         seq = [w2i_label[label] ]
        #         outputs_seq.append(seq)

        assert len(inputs_seq) == len(outputs_seq)
        # assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))

        self.w2i_char = w2i_char
        self.w2i_label = w2i_label
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)

    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            # outputs_seq_batch.append(self.outputs_seq[p].copy())
            outputs_seq = np.eye(len(self.w2i_label), dtype="int32")[self.outputs_seq[p].copy()]
            # print(outputs_seq)
            outputs_seq_batch.append(outputs_seq[0].tolist())
            # print(outputs_seq_batch)
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True

        max_seq_len = max(inputs_seq_len_batch)
        # max_seq_len = min(inputs_seq_len_batch,40)
        max_seq_len = min(max_seq_len, 100)
        inputs_seq_size_batch = zip(inputs_seq_batch, inputs_seq_len_batch)
        inputs = []
        for seq, len2 in inputs_seq_size_batch:
            if (len2 <= max_seq_len):
                seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len2))
                inputs.append(seq)
                # inputs.append(seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len2)))
            else:
                inputs.append(seq[:max_seq_len])
        # for seq in inputs_seq_batch:
        #     seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        # for seq in outputs_seq_batch:
        #     seq.extend([self.w2i_label["O"]] * (max_seq_len - len(seq)))

        return (np.array(inputs, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_batch, dtype="int32"))


if __name__ == "__main__":
    vocab_label = "vocab_char_label.txt"
    vocab_char = "vocab_char.txt"
    # load_data(path,vocab_char,vocab_label=vocab_label)
    vocab_path_char = path + vocab_char
    vocab_path_char_label = path + vocab_label
    w2i_char, i2w_char = load_vocabulary(vocab_path_char)
    w2i_label, i2w_label = load_vocabulary(vocab_path_char_label)

    print(w2i_label)
    print(i2w_label)
    path = "/Users/songdongdong/workSpace/datas/learning_data/中文文本分类/"
    train_file = "cnews.train.txt"
    test_file = "cnews.test.txt"
    val_file = "cnews.val.txt"
    data_process = DataProcessor_LSTM(path + train_file, path + train_file + "_bio", w2i_char, w2i_label, True)
    batch_data = data_process.get_batch(1000)
    print(batch_data)

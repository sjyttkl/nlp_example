# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     TextGenerate.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/11/4 09:53
   Description :   https://sg.jianshu.io/p/1bb7bad11733
==================================================
"""
import tensorflow as tf
import numpy as np

class TextGenerate:
    def __init__(self, window, corpus):
        self.window = window
        self.corpus = corpus
        self.char2id = None
        self.id2char = None
        self.char_length = 0

    def load_data(self):
        X = []
        Y = []
        # 将语料按照\n切分为句子
        corpus = self.corpus.strip().split('\n')
        # 获取所有的字符作为字典
        chrs = set(self.corpus.replace('\n', ''))
        chrs.add('UNK')
        print(corpus,len(corpus))
        print(chrs,len(chrs))
        self.char_length = len(chrs)
        self.char2id = {c: i for i, c in enumerate(chrs)}
        self.id2char = {i: c for c, i in self.char2id.items()}
        for line in corpus:
            line = line.strip()
            x = [[self.char2id[char] for char in line[i: i + self.window]] for i in range(len(line) - self.window)] #window 为1 表示 一个窗口一个窗口等产生词
            x_1 = [ line[i: i + self.window] for i in range(len(line) - self.window)]
            y = [[self.char2id[line[i + self.window]]] for i in range(len(line) - self.window)]
            y_1 = [ line[i + self.window] for i in range(len(line) - self.window)]
            X.extend(x)
            Y.extend(y)
        # 转为one-hot
        X = tf.keras.utils.to_categorical(X)
        Y = tf.keras.utils.to_categorical(Y)
        return X, Y

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)))
        model.add(tf.keras.layers.Dense(self.char_length, activation='softmax'))
        model.compile('adam', 'categorical_crossentropy')
        self.model = model

    def train_model(self, X, Y, epochs):
        self.model.fit(X, Y, epochs=epochs, verbose=1)
        self.model.save('model.model')

    def predict(self, sentence,model):
        if(model != None):
            self.model = model

        input_sentence = [self.char2id.get(char, self.char2id['UNK']) for char in sentence]
        input_sentence = tf.keras.preprocessing.sequence.pad_sequences([input_sentence], maxlen=self.window)
        input_sentence = tf.keras.utils.to_categorical(input_sentence, num_classes=self.char_length)
        predict = self.model.predict(input_sentence)
        # 本文为了方便 直接取使用最大概率的值，并非绝对，采样的方式有很多种，自行选择
        return self.id2char[np.argmax(predict)]

    def load_model(self,model_path):
        print("load model from file")
        return tf.keras.models.load_model(model_path)


if __name__ =="__main__":
    corpus = '''
    手机壳。
    手机壳。
    手机壳。
    手机壳。
    手机壳。
    手机壳。
    手机壳毛绒。
    手机壳毛绒。
    手机壳毛绒。
    手机壳毛绒。
    '''
    # 以5切分
    window = 2
    text_generate = TextGenerate(window,corpus)
    X,Y = text_generate.load_data()
    print(X.shape)
    print(Y.shape)
    text_generate.build_model()
    text_generate.train_model(X,Y,500)
    model = None
    model = text_generate.load_model("model.model")
    input_sentence = '手机'
    result = input_sentence
    #在构建语料的过程中，设置了每次只预测一个词，为了生成完成的一句话，需要进行循环预测
    while not result.endswith('。'):
        predict = text_generate.predict(input_sentence,model)
        result += predict
        input_sentence += predict
        input_sentence = input_sentence[len(input_sentence)-(window if len(input_sentence)>window else len(input_sentence)):]
        print(result)
# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     ngram.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/11/4 09:49
   Description :   https://www.jianshu.com/p/addb64c5776f
==================================================
"""

from collections import Counter
from jieba import lcut
from random import choice
#
# corpus = '''
# 这一生原本一个人，你坚持厮守成我们，却小小声牵着手在默认。
# 感动的眼神说愿意，走进我的人生。
# 进了门开了灯一家人，盼来生依然是一家人。
# 确认过眼神，我遇上对的人。
# 我挥剑转身，而鲜血如红唇。
# 前朝记忆渡红尘，伤人的不是刀刃，是你转世而来的魂。
# 青石板上的月光照进这山城，我一路的跟你轮回声，我对你用情极深。
# 谁在用琵琶弹奏一曲东风破，枫叶将故事染色，结局我看透。
# 篱笆外的古道我牵着你走过，荒烟漫草的年头，就连分手都很沉默。
# '''

corpus = '''
这一生原本一个人，你坚持厮守成我们，却小小声牵着手在默认。
感动的眼神说愿意，走进我的人生。
进了门开了灯一家人，盼来生依然是一家人。
确认过眼神，我遇上对的人。
我挥剑转身，而鲜血如红唇。
前朝记忆渡红尘，伤人的不是刀刃，是你转世而来的魂。
青石板上的月光照进这山城，我一路的跟你轮回声，我对你用情极深。
谁在用琵琶弹奏一曲东风破，枫叶将故事染色，结局我看透。
篱笆外的古道我牵着你走过，荒烟漫草的年头，就连分手都很沉默。
'''

def get_model(corpus):
    '''
    :param corpus:
    :return: { '我': Counter({'的': 2, '看透': 1}), '人生': Counter({'。': 1})}
    #           我 后面 的 出现2次 看透 出现1次
    '''
    # 将语料按照\n切分为句子
    corpus = corpus.strip().split('\n')
    # 将句子切分为词序列
    corpus = [lcut(line) for line in corpus] #分词
    # 提取所有单词
    words = [word for words in corpus for word in words]
    # 构造一个存储每个词统计的字典{'你',count(),'我':count()}
    bigram = {w: Counter() for w in words}
    # 根据语料库进行统计信息填充
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            bigram[sentence[i]][sentence[i + 1]] += 1
    return bigram

def generate_text(bigram,first_word, free=4):
    '''
    按照语言模型生成文本
    :param first_word: 提示词
    :param free: 控制范围  如果强制按照每个词后面最有可能出现的词进行生成，设置为1，需要一些灵活性，可以放宽一些
    :return:
    '''
    # 如果第一个词不在词典中 随机选一个
    if first_word not in bigram.keys():
        first_word = choice(bigram.keys())
    text = first_word
    # 将候选词按照出现概率进行降序排列
    next_words = sorted(bigram[first_word], key=lambda w: bigram[first_word][w],reverse=True)
    while True:
        if not next_words:
            break
        # 候选词前free个中随机选一个
        next_word = choice(next_words[:free])
        text += next_word
        if text.endswith('。') :
            print('生成文本：', text)
            break
        next_words = sorted(bigram[next_word], key=lambda w: bigram[next_word][w],reverse=True)

if __name__ =="__main__":
    bigram = get_model(corpus)
    first_words = ['你', '我们', '确认']
    for word in first_words:
        generate_text(bigram, word, free=1)
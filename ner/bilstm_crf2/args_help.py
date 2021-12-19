# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     args_help.py.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/12/6 13:40
   Description :  
==================================================
"""
import argparse

path = "/Users/songdongdong/workSpace/datas/learning_data/ner/1/"

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default=path+"/train.txt",help="train file")
parser.add_argument("--test_path", type=str, default=path+"/test.txt",help="test file")
parser.add_argument("--output_dir", type=str, default="checkpoints/",help="output_dir")
parser.add_argument("--vocab_file", type=str, default=path+"/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default=path+"/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=100,help="epoch")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
args = parser.parse_args()
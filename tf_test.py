# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     tf_test.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/5/27 22:36
   Description :  
==================================================
"""
import tensorflow as tf
labels = [1]
res = tf.one_hot(indices=labels,
           depth=10,
           on_value=1,
           off_value=0,
           axis=-1)
print (res )

import numpy as np
num_classes = 10
arr=[1,3]
res = np.eye(num_classes,dtype="int32")[arr]
print(res)
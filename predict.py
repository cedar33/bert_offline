import tensorflow as tf
# import ctypes
# hllDll = ctypes.WinDLL(
#     "C:\\Program Files\\NVIDIA Corporation\\NvStreamSrv\\cudart64_100.dll")
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Embeddings_bert import Embedding
import time

from text_classify import create_model

def get_voc_list(filename):
    v_list = []
    with open(filename, "r", encoding="utf-8") as f:
        v_list += f.readlines()
    v_list_2 = [v.replace("\n", "") for v in v_list]
    return v_list_2

def v_list_lookup(line_list, v_list):
    input_ids = []
    for w in line_list:
        try:
            input_ids.append(v_list.index(w))
        except ValueError:
            input_ids.append(v_list.index("。"))
    return input_ids

v_list = get_voc_list("D:\\job\\chinese_L-12_H-768_A-12\\chinese_L-12_H-768_A-12\\vocab.txt")
model = create_model(embeddings_file="D:\\job\\glove_senta\\bert_embeddings.npz", vocab_file="D:\\job\\chinese_L-12_H-768_A-12\\chinese_L-12_H-768_A-12\\vocab.txt")
model.load_weights('D:\\job\\bert_offline\\model\\')
print(time.time())
input_list = [v_list_lookup("我是一只小小小鸟", v_list)]
print(time.time())
a = model.predict(input_list)
print(time.time())

print("prediction: %s" % a[0])
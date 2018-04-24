#-*- coding: utf-8 -*-
import pickle
import argparse
import os
import sys
import csv
import find_topk_sim_func as find_topk
current_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pkl_dir',
    type=str,
    default=current_path,
    help='pickle file directory')
parser.add_argument(
    '--save_dir',
    type=str,
    default=current_path,
    help='result file save directory')
FLAGS, unparsed = parser.parse_known_args()

with open(os.path.join(FLAGS.pkl_dir, "dictionary.pkl"),"rb") as f:
    dictionary = pickle.load(f)
with open(os.path.join(FLAGS.pkl_dir, "reverse_dictionary.pkl"),"rb") as f:
    reverse_dictionary = pickle.load(f)
with open(os.path.join(FLAGS.pkl_dir, "embedding.pkl"),"rb") as f:
    embedding=pickle.load(f)
valid_examples=[]

word_list = ["경력직", "금융상담", "전기설비"]
for word in word_list:
    key = dictionary.get(word)
    valid_examples.append(key)
nearest_topk_list=find_topk.find_topk(reverse_dictionary, embedding, valid_examples)
print(nearest_topk_list)
with open(os.path.join(FLAGS.save_dir,"nearest_topk.pkl"),"wb") as f:
    pickle.dump(nearest_topk_list, f)

with open(os.path.join(FLAGS.save_dir,"nearest_topk.csv"),'w') as f:
    w = csv.writer(f)
    w.writerow(('source','target','value'))
    w.writerows([(data.source, data.target, data.value) for data in nearest_topk_list])

valid_examples=[]
for i in range(len(nearest_topk_list)):
    print(nearest_topk_list[i].target)
    word = nearest_topk_list[i].target
    key = dictionary.get(word)
    valid_examples.append(key)
nearest_topk_list2=find_topk.find_topk(reverse_dictionary, embedding, valid_examples)
print(nearest_topk_list2)
with open(os.path.join(FLAGS.save_dir,"nearest_topk2.pkl"),"wb") as f:
    pickle.dump(nearest_topk_list2, f)

with open(os.path.join(FLAGS.save_dir,"nearest_topk2.csv"),'w') as f:
    w = csv.writer(f)
    w.writerow(('source','target','value'))
    w.writerows([(data.source, data.target, data.value) for data in nearest_topk_list2])

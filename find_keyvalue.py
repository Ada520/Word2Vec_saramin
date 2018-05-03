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
parser.add_argument(
    '--step',
    type=str,
    default=100000,
    help='embedding table step')
FLAGS, unparsed = parser.parse_known_args()

with open(os.path.join(FLAGS.pkl_dir, "dictionary.pkl"),"rb") as f:
    dictionary = pickle.load(f)
with open(os.path.join(FLAGS.pkl_dir, "reverse_dictionary.pkl"),"rb") as f:
    reverse_dictionary = pickle.load(f)
try:
    with open(os.path.join(FLAGS.pkl_dir, "embedding"+FLAGS.step+".pkl"),"rb") as f:
        embedding=pickle.load(f)
        print(FLAGS.step)
except:
    with open(os.path.join(FLAGS.pkl_dir, "embedding.pkl"),"rb") as f:
        embedding = pickle.load(f)

valid_examples=[]

if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

def write_pkl_data(save_dir, step, nearest_topk_list, filename):
    with open(os.path.join(save_dir, filename + "_" + step + ".pkl"),"wb") as f:
        pickle.dump(nearest_topk_list, f)

    with open(os.path.join(save_dir, filename + "_" + step + ".csv"),"w") as f:
        w = csv.writer(f)
        w.writerow(('source','target','value'))
        w.writerows([(data.source, data.target, data.value) for data in nearest_topk_list])


print(sys.getsizeof(embedding))
embedding = embedding[:10000,:]
print(sys.getsizeof(embedding))
word_list = ["경영기획","빅데이터"]
for word in word_list:
    try:
        key = dictionary.get(word)
        valid_examples.append(key)
    except:
        continue
print(valid_examples)
nearest_topk_list_02=find_topk.find_topk(reverse_dictionary,
                                         embedding, [valid_examples[0]], 20)
nearest_topk_list_20=find_topk.find_topk(reverse_dictionary,
                                         embedding, [valid_examples[1]], 20)

write_pkl_data(FLAGS.save_dir, FLAGS.step, nearest_topk_list_02, "nearest_topk_list_02")
write_pkl_data(FLAGS.save_dir, FLAGS.step, nearest_topk_list_20, "nearest_topk_list_20")
#print(nearest_topk_list)
#with open(os.path.join(FLAGS.save_dir,"nearest_topk1_"+FLAGS.step+".pkl"),"wb") as f:
#    pickle.dump(nearest_topk_list, f)

#with open(os.path.join(FLAGS.save_dir,"nearest_topk1_"+FLAGS.step+".csv"),'w') as f:
#    w = csv.writer(f)
#    w.writerow(('source','target','value'))
#    w.writerows([(data.source, data.target, data.value) for data in nearest_topk_list])

valid_examples=[]
for i in range(len(nearest_topk_list_02)):
    #print(nearest_topk_list[i].target)
    word = nearest_topk_list_02[i].target
    key = dictionary.get(word)
    valid_examples.append(key)
nearest_topk_list_02_second=find_topk.find_topk(reverse_dictionary, embedding, valid_examples)

valid_examples=[]
for i in range(len(nearest_topk_list_20)):
    #print(nearest_topk_list[i].target)
    word = nearest_topk_list_20[i].target
    key = dictionary.get(word)
    valid_examples.append(key)
nearest_topk_list_20_second=find_topk.find_topk(reverse_dictionary, embedding, valid_examples)

write_pkl_data(FLAGS.save_dir, FLAGS.step,
               nearest_topk_list_02_second, "nearest_topk_list_02_second")
write_pkl_data(FLAGS.save_dir, FLAGS.step,
               nearest_topk_list_20_second, "nearest_topk_list_20_second")
#print(nearest_topk_list2)
#with open(os.path.join(FLAGS.save_dir,"nearest_topk2_"+FLAGS.step+".pkl"),"wb") as f:
#    pickle.dump(nearest_topk_list2, f)

#with open(os.path.join(FLAGS.save_dir,"nearest_topk2_"+FLAGS.step+".csv"),'w') as f:
#    w = csv.writer(f)
#    w.writerow(('source','target','value'))
#    w.writerows([(data.source, data.target, data.value) for data in nearest_topk_list2])

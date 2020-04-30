# Importing modules
import codecs
import numpy as np
import sys
from config import *
from ../../eval_metric.py import *

def read_map(file, map):
  # Function used to read the input files - test and validation

  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  flag = True

  d = [[]]

  i = 0
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      d.append(feats)
      flag = False
    
    elif flag == False:
      d.pop(0)
      map[i] = d
      i = i + 1
      flag = True
      d = [[]]
    
  # print(i)

def print_to_file(save_path, file_path, var):
  with open(save_path + file_path, "w") as text_file:
    text_file.write(var)

# number of runs of models to take as input
n1 = int(sys.argv[1])      # bert
n2 = int(sys.argv[2])      # roberta
n3 = int(sys.argv[3])      # xlnet

n = n1 + n2 + n3

# List of validation and test files that are input
val_list = ["" for x in range(n)]
test_list = ["" for x in range(n)]

cur_path = bert_path
for i in range(1,n1+1):
  val_list[i-1] = cur_path + 'val' + str(i) + '.txt'
  test_list[i-1] = cur_path + 'test' + str(i) + '.txt'

cur_path = roberta_path
for i in range(1,n2+1):
  val_list[n1+i-1] = cur_path + 'val' + str(i) + '.txt'
  test_list[n1+i-1] = cur_path + 'test' + str(i) + '.txt'

cur_path = xlnet_path
for i in range(1,n3+1):
  val_list[n1+n2+i-1] = cur_path + 'val' + str(i) + '.txt'
  test_list[n1+n2+i-1] = cur_path + 'val' + str(i) + '.txt'

# list of dictionaries read from files
vdict_list = [dict() for x in range(n)]
tdict_list = [dict() for x in range(n)]

for i in range(n):
  print("")
  print(val_list[i])
  read_map(val_list[i], vdict_list[i])
  read_map(test_list[i], tdict_list[i])

# Validation Data

num_m = [0, 0, 0, 0]
score_m = [0, 0, 0, 0]

all_labels = []
all_our_labels = []

dict_list = vdict_list

dummy = dict_list[0]

for i in range(val_length):
  # i iterates over the sentences in validation set
  d1 = dummy[i]

  labels = []                         # actual probability of each sentence
  our_labels = []                     # predicted probability of each sentence

  for j in range(len(d1)):
    label = (float)(d1[j][3])         # actual probability of each word
    labels.append(label)
    ls = [0.0 for x in range(n)]
    for k in range(n):
      d = dict_list[k][i][j]
      ls[k] = (float)(d[2])           # predicted probability of each word

    our_label = np.mean(ls)           # taking average of score across all files
    our_labels.append(our_label)

  all_labels.append(labels)           # true labels averaged over all input files
  all_our_labels.append(our_labels)   # predicted labels averaged over all input files

  batch_num_m, batch_score_m = match_M(all_our_labels, all_labels)

  num_m = [sum(i) for i in zip(num_m, batch_num_m)]
  score_m = [sum(i) for i in zip(score_m, batch_score_m)]

  all_labels = []
  all_our_labels = []

# Match_m score as defined in the task for m = {1,2,3,4}
m_score = [i/j for i,j in zip(score_m, num_m)]

print(m_score)
v_score = np.mean(m_score)
print(v_score)

## TEST OUTPUT

# variable used to store the final test output of the ensemble
s = ""

dict_list = tdict_list
dummy = dict_list[0]

for i in range(test_length):
  # i iterates over the sentences in test set
  d1 = dummy[i]

  our_labels = []                 # predicted probability of each sentence

  s = s + "\n"

  for j in range(len(d1)):
    ls = [0.0 for x in range(n)]
    for k in range(n):
      d = dict_list[k][i][j]
      ls[k] = (float)(d[2])       # probability score of each word

    our_label = np.mean(ls)       # predicted labels averaged over all test input files

    s = s + "{}\t{}\t{}\t".format(d1[j][0],d1[j][1],our_label) + "\n"

# Saving the ensemble of various input files
file_name = 'bert_'+str(n1)+'_roberta_'+str(n2)+'_xlnet_'+str(n3)+'.txt'
print_to_file(save_path, file_name, s)
# print_to_file(save_path, 'acc_'+file_name, str(v_score)+"\n"+str(m_score))

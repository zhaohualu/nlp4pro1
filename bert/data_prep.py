#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm_notebook
import math
import numpy as np
import sys

print(sys.argv)

DataFilename = sys.argv[1]
CVBatch = int(sys.argv[2])
#DataFilename = 'FFC_Bert_CognitiveFunctioning.txt'

prefix = 'data/'


# In[2]:

data_df = pd.read_csv(prefix + DataFilename, header=None)

for i in range(data_df.shape[0]):
    data_df.iloc[i,1] = data_df.iloc[i,1].replace("NA","")

data_df.shape
data_df.head

data_df.index
data_df.columns
data_df.values


Outcome = np.array(data_df[[0]])
OutcomeYesIdx = np.where(Outcome==2)[0]
OutcomeNoIdx = np.where(Outcome==1)[0]

CVYesIdx = np.array([1,2,3,4,5]*math.ceil(len(OutcomeYesIdx)/5))
CVYesIdx = CVYesIdx[0:len(OutcomeYesIdx)]

CVNoIdx = np.array([1,2,3,4,5]*math.ceil(len(OutcomeNoIdx)/5))
CVNoIdx = CVNoIdx[0:len(OutcomeNoIdx)]

CVIdx = np.zeros(Outcome.shape[0])
CVIdx[OutcomeYesIdx,] = CVYesIdx
CVIdx[OutcomeNoIdx] = CVNoIdx

train_df = data_df.copy()[CVIdx!=CVBatch]
test_df = data_df.copy()[CVIdx==CVBatch]

data_df.shape
train_df.shape
test_df.shape



# train_df = pd.read_csv(prefix + 'train.csv', header=None)
# train_df.head()
# In[3]:
# test_df = pd.read_csv(prefix + 'test.csv', header=None)
# test_df.head()

# In[4]:

train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)


# In[5]:


train_df2 = pd.DataFrame({
    'id':range(len(train_df)),
    'label':train_df[0],
    'alpha':['a']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
},columns=['id','label','alpha','text'])

train_df2.head()


# In[6]:


dev_df2 = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
},columns=['id','label','alpha','text'])

dev_df2.head()


# In[7]:


train_df2.to_csv('data/train.tsv', sep='\t', index=False, header=False, columns=train_df2.columns)
dev_df2.to_csv('data/dev.tsv', sep='\t', index=False, header=False, columns=dev_df2.columns)


# In[ ]:





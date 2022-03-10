#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from re import S, sub 
import shutil 
from glob import glob


# In[22]:


for folder in glob('./results/*'):
    print(folder)
    for file in glob(folder+'/*.png'):
        if os.path.isdir(file):
            continue
        print(file.split('_'))
        res = file.split('_')[-5]
        
        if not os.path.isdir(folder+f'/{res}/'):
            os.mkdir(folder+f'/{res}/')
        shutil.move(file,folder+f'/{res}/')


# In[ ]:





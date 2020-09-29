# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:20:18 2020

@author: Kaja Amalie
"""


import os
import pandas as pd
import numpy as np

os.chdir(r'C:\Users\Kaja Amalie\Documents\Kaja\project_ML')
df2 = pd.read_csv('item_categories.csv')
df3 = pd.read_csv('items.csv')

categories = df2['item_category_name']
items = df3['item_name']

#create a function to test if the translation works
from googletrans import Translator
def translator(data):
    trans = Translator()
    lis = ''
    for line in data:
        line2 = trans.translate(line, sr = 'rus', dest= 'en')
        print(line2.text)
        lis += line2.text
    return(lis)



#create english categories
categories_name = []
trans = Translator()
for line in categories:
    line2 = trans.translate(line, sr = 'rus', dest= 'en')
    print(line2.text)
    categories_name.append(line2.text)

categories_ar = np.c_[categories_name]

df2['item_category_name'] = categories_ar
df2 = df2[['item_category_name', 'item_category_id']]
df2.to_csv('categories.csv', index = False)
      
items_name = []
trans = Translator()
for line in items:
    line2 = trans.translate(line, sr = 'rus', dest= 'en')
    print(line2.text)
    items_name.append(line2.text)
    
items_ar = np.c_[items_name]
del df3['item_name']
df3['item_name'] = items_ar
df3 = df3[['item_name', 'item_id', 'item_category_id']]
df3.to_csv('items.csv', index = False)







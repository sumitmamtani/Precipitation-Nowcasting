import os, os.path, glob
from scipy.stats import iqr
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import io
import imageio
import cv2
import sys
from ipywidgets import widgets, HBox

file_counter = 0
global_ans = []
gc=0
for foldername in os.listdir("nh_radar_comp_echo/"):
    if foldername=='.DS_Store':
        continue
    gc+=1
    list = os.listdir("nh_radar_comp_echo/"+foldername) # dir is your directory path
    number_files = len(list)
    if number_files<240:
        continue
    file_counter+=1
    ls = []
    counter = 0 
    for filename in sorted(os.listdir("nh_radar_comp_echo/"+foldername)):
        if counter>=240:
            break
        counter+=1
        img = cv2.imread("nh_radar_comp_echo/"+foldername+"/"+filename) 
        img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
        np_img = np.array(img)
        ls.append(np_img)
    f = np.asarray(ls)
    global_ans.append(f)
Radar2 = np.asarray(global_ans)
np.save('new_radar_20_128fr_cleaned2.npy', Radar2)

Radar2 = np.load('new_radar_20_128fr_cleaned2.npy')
Radar2.resize((5784, 20, 128, 128))

print(np.max(Radar2))
print(np.min(Radar2))


op = []
for i in range(0, 5725):
    for j in range(0, 20):
        op.append(np.sum(Radar2[i][j]))
        
st = np.asarray(op)
print(st.shape)

n25 = np.percentile(st, 25)
n75 = np.percentile(st, 75)

print(n25, n75)

gl = []
maxp = -1
minp = sys.maxsize
for i in range(0, 5725):
    no_bad = 0;
    for j in range(0, 20):
        sum_fr = np.sum(Radar2[i][j])
        if sum_fr<n25 or sum_fr>n75:
            no_bad+=1
    if no_bad<=10:
        gl.append(Radar2[i])
Radar = np.asarray(gl)

np.random.shuffle(Radar)
print(Radar.shape)
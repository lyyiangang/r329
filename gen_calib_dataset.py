import numpy as np
import sys
import os
import cv2

img_dir='./dataset/img/'
label_file='./dataset/label.txt'

input_height=224
input_width=224
mean = np.array([0.485, 0.456, 0.406], np.float32)
var = np.array([0.229, 0.224, 0.225], np.float32)

imgs = []
labels = []
with open(label_file, 'r') as fid:
    for line in fid:
        filename, label = line.rstrip('\n').split(' ')
        labels.append(int(label))
        img = cv2.imread(os.path.join(img_dir, filename)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ToTensor and Normalize
        img = cv2.resize(img, (input_width, input_height)).astype(np.float32) / 255
        norm_img = (img - mean) / var
        imgs.append(norm_img)

imgs = np.asarray(imgs)
labels = np.asarray(labels)
out_label = 'dataset/label.npy'
out_dataset = 'dataset/dataset.npy'
print(f'writing {out_label} and {out_dataset}')
np.save(out_label, labels)
np.save(out_dataset, imgs)
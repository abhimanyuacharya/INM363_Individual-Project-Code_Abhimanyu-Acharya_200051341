# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 23:48:53 2022

@author: Abhimanyu
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

#Function to import labels 
def loadandProcessImagesandLabels(path_to_data,folderName,labels_path,labelFileName,numImages):
    image_list = []
    names = []
    labels = []

    
    with open(str(labels_path + labelFileName + '.txt'), 'r') as f:
        for line in f.readlines():
                tokens = line.split(' ')
                names.append(str(tokens[0]))
                labels.append(int(tokens[1]))          
                
    labeldf = pd.DataFrame({'Names':names, 'Labels':labels})           
    labeldf["Names"]= labeldf["Names"].str.split('.').str[0]   
    labeldf = labeldf.sort_values(by=['Names'])    

    #Load images by name
    for name in labeldf["Names"]:
        img = cv2.imread(str(path_to_data + folderName + '/' + name +  '_aligned.jpg'),3)
        image_list.append(img)   
    image_array = np.array(image_list)
       
    #train or test labels
    if folderName == 'train':
        print("Full Trianing Data: {n} images of size {s} x {s}".format(n = image_array.shape[0],s = image_array.shape[1]))
    if folderName == 'test':
        print("Test Data: {n} images of size {s} x {s}".format(n = image_array.shape[0],s = image_array.shape[1]))
    
    #show the images
    fig, axes = plt.subplots(nrows=int(numImages/5), ncols=5, figsize=(12, 12), sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(numImages):
        ax[i].imshow(cv2.cvtColor(image_array[i, :, :], cv2.COLOR_BGR2RGB))
        ax[i].set_title(f'Label: {Label2Emotion(labels[i])} ({labels[i]})',fontweight="bold")
        ax[i].set_axis_off()     
    fig.tight_layout()
    plt.show()    
    
    return image_list, labels

def Label2Emotion(label):
  if label == 1:
      x = 'Surprise'
  if label == 2:
      x = 'Fear'
  if label == 3:
      x = 'Disgust'
  if label == 4:
      x = 'Happiness'
  if label == 5:
      x = 'Sadness'
  if label == 6:
      x = 'Anger'
  if label == 7:
      x = 'Neutral'

  return x

def Label2EmotionCNN(label):
  if label == 0:
      x = 'Surprise'
  if label == 1:
      x = 'Fear'
  if label == 2:
      x = 'Disgust'
  if label == 3:
      x = 'Happiness'
  if label == 4:
      x = 'Sadness'
  if label == 5:
      x = 'Anger'
  if label == 6:
      x = 'Neutral'

  return x
        

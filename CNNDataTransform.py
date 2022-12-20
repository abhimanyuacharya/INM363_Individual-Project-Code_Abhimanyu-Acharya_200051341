# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:17:00 2022

@author: Abhimanyu
"""
#Adapted from various responses on Stackexchange and CV Lab 8
class dataTransform():
  def __init__(self, data, targets, transform=None):
    self.data = data
    self.targets = targets

    self.transform = transform

  def __getitem__(self,index):
    x = self.data[index]
    y = self.targets[index]

    if self.transform:
      x = self.transform(x)
    
    return x, y

  def __len__(self):
    return len(self.data)


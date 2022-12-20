# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:47:07 2022
@author: Abhimanyu Acharya
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import cv2


def AWA2DataLoader(data_dir, data_type, Resnet101Type = None,  animal_attributes = True, taxonomy = True):
    
    # Import datasets 
    #Class Names
    classes_df = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/classes.txt", sep = '\t',header = None, names = ['labels','classes'])
    class_names = np.sort(classes_df['classes'].tolist())
    #print(class_labels)
    
    #Image names
    image_names_df = pd.read_csv(data_dir + "AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-filenames.txt", sep = " ",header = None, names = ['filenames'])
    image_names = image_names_df['filenames'].tolist()
    #print(image_names)
    
    #Animal Attribute Name
    predicates_df = pd.read_csv(data_dir +"AwA2-data/Animals_with_Attributes2/predicates.txt", sep = '\t',header = None, names = ['predicates'])
    predicate_list = predicates_df['predicates'].tolist()
    
    #Image Labels
    Resnet101_labels_df = pd.read_csv("C:/Dissertation/Data/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt", sep = " ",header = None, names = ['labels'] )
    
    #Binary attributes
    predicateMatrixBinary_df = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/predicate-matrix-binary.txt", sep = " ", header = None, names = predicate_list)
    
    #Continous Attributes
    predicateMatrixContinous_df = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/predicate-matrix-continuous_norm12.csv", header = None, names = predicate_list, dtype=float)
    
    # Concat predicate types with binary and continous values 
    predicate_df = pd.concat([classes_df, predicateMatrixBinary_df, predicateMatrixContinous_df], axis=1)
    
    if data_type == "ResNet101-Features":
        #Import Resnet101 features 
        st = time.time()
        
        if Resnet101Type == 1:
            print('Using Resnet101 Features provided with data')
            Resnet101_features_df = pd.read_csv(data_dir +"/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt", sep = " ", header = None)
            #concat filenames Resnet10 features & class labels
            Imagedata_df = pd.concat([image_names_df, Resnet101_labels_df, Resnet101_features_df], axis=1)
                
            #merge with class names
            Final_Images_df = pd.merge(Imagedata_df,classes_df,  how='left', on='labels')
            Final_Images_df.drop(['labels'], axis = 1, inplace = True)
            
                        
        elif Resnet101Type == 2: 
            print('Using Resnet101 Features extracted by Abhimanyu Acharya using a pretrainied Resnet101')            
            Final_Images_df = pd.read_csv(data_dir +"/Processed Data/Extracted_ResNet101_features.csv")
            Final_Images_df.drop(['Unnamed: 0'], 1, inplace = True)
            #Final_Images_df.drop(['labels'], axis = 1, inplace = True)
            
        else:
            pass           
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')    
    elif data_type == "Images":
        Images = [] #Empty list to add loaded images to   
        
        #loop trhough folder and load image if file name matches
        st = time.time()
        for class_name in class_names: 
            path = os.path.join(data_dir + "AwA2-data/Animals_with_Attributes2/JPEGImages", class_name)
            for image in os.listdir(path):
                if image in image_names:
                    img_array = cv2.imread(os.path.join(path, image)) 
                    image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #convert BGR to RGB format
                    resized_array = cv2.resize(image_rgb, (224, 224)) # Reshaping images to preferred size
                    Images.append([resized_array, class_name])
                    #print(image, ',', class_name,'- Image Loaded')
                else:
                    print(image, 'Image not found')
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')   
                           
        #convert to numpy array 
        images = np.asarray([item[0] for item in Images])
        classes = np.asarray([item[1] for item in Images])
        
        #convert arrays to pandas dataframe to combine
        images_df = pd.DataFrame(images.reshape(-1,224*224*3))
        class_df = pd.DataFrame(classes, columns = ['classes'])

        Final_Images_df = pd.concat([image_names_df, class_df,  images_df], axis = 1)
        
        #print 25 random images
        randomimages = random.sample(list(Images), 25)
        printimages = np.asarray([item[0] for item in randomimages])
        printclasses = np.asarray([item[1] for item in randomimages])
        
        #show the images
        fig, axes = plt.subplots(nrows= 5, ncols=5, figsize=(12, 12), sharex=True, sharey=True)
        ax = axes.ravel()
        for i in range(25):
            ax[i].imshow(printimages[i])
            ax[i].set_title(f'Label: {printclasses[i]}')
            ax[i].set_axis_off()     
        fig.tight_layout()
        plt.show()          
        
    if animal_attributes is True:
            
      # Merge with Image/Resnet101 features
      #Train Test split
      Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
      print("Animal Attributes Included")
      
     
      if taxonomy is True:
          taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
          taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
      
          #convert categorical data to dummy variables
          taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19])
          taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
          taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
          
          #Merge with Image/Resnet101 features
          Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
          
          #Merge with animal attribute data
          predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
          print('Animal Taxonomy Included')
          
      else:
          print('No Animal Taxonomy Included')
          pass
  
    else:
      print("No Animal Attributes Included")
      pass
     
    return Final_Images_df, predicate_df
    

def TestTrainValsplitdata_dir(data_dir, dataframe, data_type, taxonomy = False):
    # Load indices for predefined split v.2.0
    trainval_index = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/Trainval split.csv", header = None, names = ['index'])
    trainval_index_list = trainval_index['index'].to_list()
    trainval_index_list = [x - 1 for x in trainval_index_list] # Provided index is from Matlab and starts with 1
    
    train_index = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/Train split.csv", header = None, names = ['index'])
    train_index_list = train_index['index'].to_list()
    train_index_list = [x - 1 for x in train_index_list] # Provided index is from Matlab and starts with 1
    
    val_index = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/Val split.csv", header = None, names = ['index'])
    val_index_list = val_index['index'].to_list()
    val_index_list = [x - 1 for x in val_index_list] # Provided index is from Matlab and starts with 1
    
    testseen_index = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/Test seen split.csv", header = None, names = ['index'])
    testseen_index_list = testseen_index['index'].to_list()
    testseen_index_list = [x - 1 for x in testseen_index_list] # Provided index is from Matlab and starts with 1
    
    testunseen_index = pd.read_csv(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/Test unseen split.csv", header = None, names = ['index'])
    testunseen_index_list = testunseen_index['index'].to_list()
    testunseen_index_list = [x - 1 for x in testunseen_index_list] # Provided index is from Matlab and starts with 1
    
    #Train Validation Split
    trainval_df = dataframe[dataframe.index.isin(trainval_index_list)]
    #print('\033[1m','\033[91m',"Training + Validation",'\033[0m')
    #print('\033[1m',"Train+ Validation Classes",'\033[0m','\n', trainval_df.classes.unique(),'\n')
          
    #Predefined Train - Validation Split 01
    trainclasses1_classes_file = open(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/trainclasses1.txt", "r")
    # reading the file
    trainclasses1_classes_names = trainclasses1_classes_file.read()
    # replacing end of line('/n') with ' ' and
    trainclasses1_classes_names_list = trainclasses1_classes_names.replace('\n', ' ').split(" ")
    # split in to train and val data
    TrainData01 = trainval_df[trainval_df['classes'].isin(trainclasses1_classes_names_list)]
    ValData01 = trainval_df[~trainval_df['classes'].isin(trainclasses1_classes_names_list)]
    # printing the data
    print('\033[1m','\033[91m',"Training - Validation Split 01",'\033[0m')
    print('\033[1m',"Train Classes",'\033[0m','\n', len(TrainData01.classes),'\n', TrainData01.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', len(ValData01.classes),'\n', ValData01.classes.unique(),'\n')
        
    #Predefined Train - Validation Split 02
    trainclasses2_classes_file = open(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/trainclasses2.txt", "r")
    # reading the file
    trainclasses2_classes_names = trainclasses2_classes_file.read()
    # replacing end of line('/n') with ' ' and
    trainclasses2_classes_names_list = trainclasses2_classes_names.replace('\n', ' ').split(" ")
    # split in to train and val data
    TrainData02 = trainval_df[trainval_df['classes'].isin(trainclasses2_classes_names_list)]
    ValData02 = trainval_df[~trainval_df['classes'].isin(trainclasses2_classes_names_list)]
    # printing the data
    print('\033[1m','\033[91m',"Training - Validation Split 02",'\033[0m')
    print('\033[1m',"Train Classes",'\033[0m','\n', len(TrainData02.classes),'\n', TrainData02.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', len(ValData02.classes),'\n', ValData02.classes.unique(),'\n')
        
    #Predefined Train - Validation Split 03
    trainclasses3_classes_file = open(data_dir + "AwA2-data/Animals_with_Attributes2/Updated Splits/trainclasses3.txt", "r")
    # reading the file
    trainclasses3_classes_names = trainclasses3_classes_file.read()
    # replacing end of line('/n') with ' ' and
    trainclasses3_classes_names_list = trainclasses3_classes_names.replace('\n', ' ').split(" ")
    # split in to train and val data
    TrainData03 = trainval_df[trainval_df['classes'].isin(trainclasses3_classes_names_list)]
    ValData03 = trainval_df[~trainval_df['classes'].isin(trainclasses3_classes_names_list)]
    # printing the data
    print('\033[1m','\033[91m',"Training - Validation Split 03",'\033[0m')
    print('\033[1m',"Train Classes",'\033[0m','\n', len(TrainData03.classes),'\n', TrainData03.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', len(ValData03.classes),'\n', ValData03.classes.unique(),'\n')
    
    #Train Split
    TrainData = dataframe[dataframe.index.isin(train_index_list)]
    print('\033[1m','\033[91m',"Training",'\033[0m')
    print('\033[1m',"Train Classes",'\033[0m','\n', len(TrainData.classes),'\n', TrainData.classes.unique(),'\n')
    
    #Val split
    ValData = dataframe[dataframe.index.isin(val_index_list)]
    print('\033[1m','\033[91m',"Validation",'\033[0m')
    print('\033[1m',"Validation Classes",'\033[0m','\n', len(ValData.classes),'\n', ValData.classes.unique(),'\n')
    
    #Seen Test 
    Seen_test_df = dataframe[dataframe.index.isin(testseen_index_list)]
    print('\033[1m','\033[91m',"Seen Test",'\033[0m')
    print('\033[1m',"Seen Test Classes",'\033[0m','\n', len(Seen_test_df.classes),'\n', Seen_test_df.classes.unique(),'\n')
    
    #Unseen Test
    Unseen_test_df = dataframe[dataframe.index.isin(testunseen_index_list)]
    print('\033[1m','\033[91m',"UnSeen Test",'\033[0m')
    print('\033[1m',"UnSeen Test Classes",'\033[0m','\n', len(Unseen_test_df.classes),'\n', Unseen_test_df.classes.unique(),'\n')
    
        
    if data_type == 'Images' and taxonomy is False:
    # Define X, y variable for training
        X1 = TrainData01.iloc[:,2:150530].values
        X1 = X1.reshape(-1, 224, 224,3) #reshae back into an image
        y1 = TrainData01.iloc[:,150531:150616]
        
        # Pre defined Test split 02
        X2 = TrainData02.iloc[:,2:150530].values
        X2 = X2.reshape(-1, 224, 224,3) #reshae back into an image
        y2 = TrainData02.iloc[:,150531:150616]
        
        # Pre defined Test split 03
        X3 = TrainData03.iloc[:,2:150530].values
        X3 = X3.reshape(-1, 224, 224,3) #reshae back into an image
        y3 = TrainData03.iloc[:,150531:150616]
        
        X_finaltrain = trainval_df.iloc[:,2:150530].values
        X_finaltrain.reshape(-1, 224, 224,3) #reshae back into an image
        y_finaltrain = trainval_df.iloc[:,150531:150616]
        
        #print 5 random images
        randomimages = random.sample(list(X1), 5)
                
        #show the images
        fig, axes = plt.subplots(nrows= 1, ncols=5, figsize=(12, 12), sharex=True, sharey=True)
        ax = axes.ravel()
        for i in range(5):
            ax[i].imshow(randomimages[i])
            ax[i].set_axis_off()     
        fig.tight_layout()
        plt.show()     
        
    elif data_type == 'Images' and taxonomy is True:
    # Define X, y variable for training
        X1 = TrainData01.iloc[:,2:150530].values
        X1 = X1.reshape(-1, 224, 224,3) #reshae back into an image
        Y1 = TrainData01.iloc[:,150531:]
        y1 = pd.merge(Y1.iloc[:, 0:85], Y1.iloc[:,170:], on = Y1.index)
        y1.drop(['key_0'], 1, inplace = True)
        
        # Pre defined Test split 02
        X2 = TrainData02.iloc[:,2:150530].values
        X2 = X2.reshape(-1, 224, 224,3) #reshae back into an image
        Y2 = TrainData02.iloc[:,150531:]
        y2 = pd.merge(Y2.iloc[:, 0:85], Y2.iloc[:,170:], on = Y2.index)
        y2.drop(['key_0'], 1, inplace = True)
        
        # Pre defined Test split 03
        X3 = TrainData03.iloc[:,2:150530].values
        X3 = X3.reshape(-1, 224, 224,3) #reshae back into an image
        Y3 = TrainData03.iloc[:,150531:]
        y3 = pd.merge(Y3.iloc[:, 0:85], Y3.iloc[:,170:], on = Y3.index)
        y3.drop(['key_0'], 1, inplace = True)
        
        X_finaltrain = trainval_df.iloc[:,2:150530].values
        X_finaltrain = X_finaltrain.reshape(-1, 224, 224,3) #reshae back into an image
        Y_finaltrain = trainval_df.iloc[:,150531:]
        y_finaltrain = pd.merge(Y_finaltrain.iloc[:, 0:85], Y_finaltrain.iloc[:,170:], on = Y_finaltrain.index)
        y_finaltrain.drop(['key_0'], 1, inplace = True)
        
        #print 5 random images
        randomimages = random.sample(list(X1), 5)
                
        #show the images
        fig, axes = plt.subplots(nrows= 1, ncols=5, figsize=(12, 12), sharex=True, sharey=True)
        ax = axes.ravel()
        for i in range(5):
            ax[i].imshow(randomimages[i])
            ax[i].set_axis_off()     
        fig.tight_layout()
        plt.show()   
        
    elif data_type == 'ResNet101-Features' and taxonomy is False:
        # Define X, y variable for training
        # Pre defined Test split 01
        X1 = TrainData01.iloc[:,2:2050]
        y1 = TrainData01.iloc[:,2051:2136]
              
        
        # Pre defined Test split 02
        X2 = TrainData02.iloc[:,2:2050]
        y2 = TrainData02.iloc[:,2051:2136]
                
        # Pre defined Test split 03
        X3 = TrainData03.iloc[:,2:2050]
        y3 = TrainData03.iloc[:,2051:2136]
              
        # Pre defined Test split 03
        X_finaltrain = trainval_df.iloc[:,2:2050]
        y_finaltrain = trainval_df.iloc[:,2051:2136]
        
    elif data_type == 'ResNet101-Features' and taxonomy is True:
        # Define X, y variable for training
    
        X1 = TrainData01.iloc[:,2:2050]
        Y1 = TrainData01.iloc[:,2051:]
        y1 = pd.merge(Y1.iloc[:, 0:85], Y1.iloc[:,170:], on = Y1.index)
        y1.drop(['key_0'], 1, inplace = True)
                          
        # Pre defined Test split 02
        X2 = TrainData02.iloc[:,2:2050]
        Y2 = TrainData02.iloc[:,2051:]
        y2 = pd.merge(Y2.iloc[:, 0:85], Y2.iloc[:,170:], on = Y2.index)
        y2.drop(['key_0'], 1, inplace = True)
                  
        # Pre defined Test split 03
        X3 = TrainData03.iloc[:,2:2050]
        Y3 = TrainData03.iloc[:,2051:]
        y3 = pd.merge(Y3.iloc[:, 0:85], Y3.iloc[:,170:], on = Y3.index)
        y3.drop(['key_0'], 1, inplace = True)
                
        # Pre defined Test split 03
        X_finaltrain = trainval_df.iloc[:,2:2050]
        Y_finaltrain = trainval_df.iloc[:,2051:]
        y_finaltrain = pd.merge(Y_finaltrain.iloc[:, 0:85], Y_finaltrain.iloc[:,170:], on = Y_finaltrain.index)
        y_finaltrain.drop(['key_0'], 1, inplace = True)
                
                    
    return X1, X2, X3, X_finaltrain, y1, y2, y3, y_finaltrain
Footer
© 2022 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Docs
    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

Dissertation/Utility_functions.py at main · abhimanyuacharya/Dissertation
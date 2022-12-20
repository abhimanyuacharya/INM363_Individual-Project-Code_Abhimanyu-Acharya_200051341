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


def AWA2DataLoader(data_dir, data_type, Resnet101Type,  animal_attributes, VectorType = None):
    
    # Import datasets 
    #Class Names
    st = time.time()
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
           
        
    elif data_type == "Images":
        Images = [] #Empty list to add loaded images to   
        
        #loop trhough folder and load image if file name matches
        for class_name in class_names: 
            path = os.path.join(data_dir + "AwA2-data/Animals_with_Attributes2/JPEGImages", class_name)
            for image in os.listdir(path):
                if image in image_names:
                    img_array = cv2.imread(os.path.join(path, image)) 
                    image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #convert BGR to RGB format
                    resized_array = cv2.resize(image_rgb, (224, 224)) # Reshaping images to preferred size
                    Images.append([resized_array, class_name])
                    print(image, ',', class_name,'- Image Loaded')
                else:
                    print(image, 'Image not found')
                                 
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
        
    if animal_attributes == 'AwA2':
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
                        
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
        
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df.iloc[:,:87],taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')

        
    elif animal_attributes == 'Custom Animal Attribute KG':
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_binary_vectors.csv")
            KG_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_continous_vectors.csv")
            KG_embedding_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            
        KG_embedding_df.rename(columns={"index": "classes"}, inplace = True)
        classes_col = KG_embedding_df.pop('classes')
        KG_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,KG_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,KG_embedding_df,  how='left', on='classes')
        
        print("Custom KG Animal Attributes Included") 
        
        
    elif animal_attributes == 'Custom KG Extended 01':
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_binary_vectors_extended01.csv")
            KG_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_continous_vectors_extended01.csv")
            KG_embedding_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            
        KG_embedding_df.rename(columns={"index": "classes"}, inplace = True)
        classes_col = KG_embedding_df.pop('classes')
        KG_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,KG_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,KG_embedding_df,  how='left', on='classes')
        
        print("Custom KG Extended 01 Included") 
        
    elif animal_attributes == 'Custom KG Extended 02':
         if VectorType == 'Binary':
             predicate_df = predicate_df.iloc[:,0:87]
             Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
             print("Animal Attributes Included - Binary Vector") 
             
         elif VectorType == 'Continous':
             predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
             Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
             print("Animal Attributes Included - Continous Vector") 
             
         taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
         taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
     
         #convert categorical data to dummy variables
         taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
         taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
         taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
         
         if VectorType == 'Binary':
             KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_binary_vectors_extended02.csv")
             KG_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
             
         elif VectorType == 'Continous':    
             KG_embedding_df = pd.read_csv(data_dir+ "Processed Data/Custom_Animal_Attribute_KG_continous_vectors_extended02.csv")
             KG_embedding_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
             
         KG_embedding_df.rename(columns={"index": "classes"}, inplace = True)
         classes_col = KG_embedding_df.pop('classes')
         KG_embedding_df.insert(0, 'classes', classes_col)  
         
         #Merge with Image/Resnet101 features
         Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
         Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
         Final_Images_df = pd.merge(Final_Images_df,KG_embedding_df,  how='left', on='classes')
         
         #Merge with animal attribute data
         predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
         predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
         predicate_df = pd.merge(predicate_df,KG_embedding_df,  how='left', on='classes')
         
         print("Custom KG Extnded 02 Included")     
               
    elif animal_attributes == 'Word2Vec':   
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Word2Vec_binary_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Word2Vec_continous_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','word2vec_names','labels'], axis = 1, inplace = True)
                                 
        classes_col = Word2Vec_embedding_df.pop('classes')
        Word2Vec_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        print("Word2Vec Animal Attributes Included") 
   
    elif animal_attributes == 'Word2Vec Extended':   
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Word2Vec_binary_vectors_extended.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Word2Vec_continous_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','word2vec_names','labels'], axis = 1, inplace = True)
                                 
        classes_col = Word2Vec_embedding_df.pop('classes')
        Word2Vec_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        print("Word2Vec Animal Attributes Extended Included") 
        
    elif animal_attributes == 'Dbnary':   
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/DBnary_binary_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/DBnary_continous_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','word2vec_names','labels'], axis = 1, inplace = True)
                                 
        classes_col = Word2Vec_embedding_df.pop('classes')
        Word2Vec_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        print("Dbnary Animal Attributes Included") 
      
    elif animal_attributes == 'DBpedia':   
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/DBpedia_binary_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/DBpedia_continous_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','word2vec_names','labels'], axis = 1, inplace = True)
                                 
        classes_col = Word2Vec_embedding_df.pop('classes')
        Word2Vec_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        print("DBpedia Animal Attributes Included")  
        
    
    elif animal_attributes == 'WordNet':   
        if VectorType == 'Binary':
            predicate_df = predicate_df.iloc[:,0:87]
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Binary Vector") 
            
        elif VectorType == 'Continous':
            predicate_df = pd.concat([predicate_df['classes'], predicate_df.iloc[:,87:172]], axis =1)
            Final_Images_df = pd.merge(Final_Images_df,predicate_df,  how='left', on='classes')
            print("Animal Attributes Included - Continous Vector") 
            
        taxonomy_df = pd.read_csv(data_dir + "Mammal Taxonomy/Mammal Taxonomy_noduplicates.csv")
        taxonomy_df.drop(['subgenus','biogeographicRealm','CMW_sciName'], axis = 1, inplace = True)
    
        #convert categorical data to dummy variables
        taxonomy_dummy_df = pd.get_dummies(taxonomy_df.iloc[:,5:19], drop_first=True)
        taxonomy_final_df = pd.concat([taxonomy_df['animal_class'],taxonomy_dummy_df], axis = 1)
        taxonomy_final_df.rename(columns={"animal_class": "classes"}, inplace = True)
        
        if VectorType == 'Binary':
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Wordnet_binary_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','order','family','K_mean'], axis = 1, inplace = True)
            
        elif VectorType == 'Continous':    
            Word2Vec_embedding_df = pd.read_csv(data_dir+ "Processed Data/Wordnet_continous_vectors.csv")
            Word2Vec_embedding_df.drop(['Unnamed: 0','word2vec_names','labels'], axis = 1, inplace = True)
                                 
        classes_col = Word2Vec_embedding_df.pop('classes')
        Word2Vec_embedding_df.insert(0, 'classes', classes_col)  
        
        #Merge with Image/Resnet101 features
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_final_df,  how='left', on='classes')
        Final_Images_df = pd.merge(Final_Images_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        Final_Images_df = pd.merge(Final_Images_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        #Merge with animal attribute data
        predicate_df = pd.merge(predicate_df,taxonomy_final_df,  how='left', on='classes')
        predicate_df = pd.merge(predicate_df,taxonomy_df,  left_on='classes', right_on='animal_class')
        predicate_df = pd.merge(predicate_df,Word2Vec_embedding_df,  how='left', on='classes')
        
        print("WordNet Animal Attributes Included")  
         
    else:
      print("No Animal Attributes Included")
      pass                
    
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds') 
    
    return Final_Images_df, predicate_df
    

def TestTrainValsplitdata_dir(data_dir, dataframe):
    st = time.time()
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
    print('\033[1m',"Train Classes",'\033[0m','\n', 'Number of Classes', len(TrainData01.classes.unique()),'\n', TrainData01.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', 'Number of Classes', len(ValData01.classes.unique()),'\n', ValData01.classes.unique(),'\n')
        
        
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
    print('\033[1m',"Train Classes",'\033[0m','\n', 'Number of Classes', len(TrainData02.classes.unique()),'\n', TrainData02.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', 'Number of Classes', len(ValData02.classes.unique()),'\n', ValData02.classes.unique(),'\n')
        
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
    print('\033[1m',"Train Classes",'\033[0m','\n', 'Number of Classes', len(TrainData03.classes.unique()),'\n', TrainData03.classes.unique(),'\n')
    print('\033[1m',"Validation Classes",'\033[0m','\n', 'Number of Classes', len(ValData03.classes.unique()),'\n', ValData03.classes.unique(),'\n')
    
    #Train Split
    TrainData = dataframe[dataframe.index.isin(train_index_list)]
    print('\033[1m','\033[91m',"Training",'\033[0m')
    print('\033[1m',"Train Classes",'\033[0m','\n', 'Number of Classes', len(TrainData.classes.unique()),'\n', TrainData.classes.unique(),'\n')
    
    #Val split
    ValData = dataframe[dataframe.index.isin(val_index_list)]
    print('\033[1m','\033[91m',"Validation",'\033[0m')
    print('\033[1m',"Validation Classes",'\033[0m','\n', 'Number of Classes', len(ValData.classes.unique()),'\n', ValData.classes.unique(),'\n')
    
    #Seen Test 
    Seen_test_df = dataframe[dataframe.index.isin(testseen_index_list)]
    print('\033[1m','\033[91m',"Seen Test",'\033[0m')
    print('\033[1m',"Seen Test Classes",'\033[0m','\n', 'Number of Classes', len(Seen_test_df.classes.unique()),'\n', Seen_test_df.classes.unique(),'\n')
    
    #Unseen Test
    Unseen_test_df = dataframe[dataframe.index.isin(testunseen_index_list)]
    print('\033[1m','\033[91m',"UnSeen Test",'\033[0m')
    print('\033[1m',"UnSeen Test Classes",'\033[0m','\n', 'Number of Classes', len(Unseen_test_df.classes.unique()),'\n', Unseen_test_df.classes.unique(),'\n')
    
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds') 
    
    return trainval_df, TrainData01, ValData01, TrainData02, ValData02, TrainData03, ValData03, Seen_test_df, Unseen_test_df
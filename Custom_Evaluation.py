# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:04:49 2022

@author: Abhimanyu Acharya
"""
import numpy as np
import pandas as pd

def per_class_top_k_accuracy_score(prob, labels, unseen_class_df, animallabels, Train = True):
    
    for k in range(2,6):
        #get index of top k values
        CZSL_label_prediction_proba = np.argpartition(prob, -k, axis=1)[:, -k:]
        #print(prob)
        #print(CZSL_label_prediction_proba)
        #print(labels)
        #replace index values with corresponding class labels
        if Train == True:
            CZSL_label_prediction_proba = np.select([CZSL_label_prediction_proba == 0, CZSL_label_prediction_proba == 1
                      , CZSL_label_prediction_proba == 2, CZSL_label_prediction_proba == 3, CZSL_label_prediction_proba == 4
                      , CZSL_label_prediction_proba == 5, CZSL_label_prediction_proba == 6, CZSL_label_prediction_proba == 7
                      , CZSL_label_prediction_proba == 8, CZSL_label_prediction_proba == 9, CZSL_label_prediction_proba == 10
                      , CZSL_label_prediction_proba == 11, CZSL_label_prediction_proba == 12],
                        [labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8], 
                         labels[9], labels[10], labels[11], labels[12]], CZSL_label_prediction_proba)
        else:
            CZSL_label_prediction_proba = np.select([CZSL_label_prediction_proba == 0, CZSL_label_prediction_proba == 1, CZSL_label_prediction_proba == 2, CZSL_label_prediction_proba == 3
                      , CZSL_label_prediction_proba == 4, CZSL_label_prediction_proba == 5, CZSL_label_prediction_proba == 6, CZSL_label_prediction_proba == 7
                      , CZSL_label_prediction_proba == 8, CZSL_label_prediction_proba == 9], [labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]
                      , labels[6], labels[7], labels[8], labels[9]], CZSL_label_prediction_proba)

        #concat predicted top k labels with true labels
        Unseenlabelsdf = unseen_class_df.copy()
        Unseenlabelsdf.reset_index(drop=True, inplace=True)
        CZSL_label_prediction_proba_df = pd.DataFrame(CZSL_label_prediction_proba)
        topkdf = pd.concat([Unseenlabelsdf.labels,CZSL_label_prediction_proba_df], ignore_index=True,axis=1)
        #print(topkdf)
        #update new column to get correct top k predictions per class
        if k == 1:
            topkdf['top_k_pred'] = np.where((topkdf[1] == topkdf[0]) , np.int(1), 0) 
        elif k ==2:
            topkdf['top_k_pred'] = np.where((topkdf[1] == topkdf[0]) | (topkdf[2] == topkdf[0]), np.int(1), 0) 
        elif k == 3:
            topkdf['top_k_pred'] = np.where((topkdf[1] == topkdf[0]) | (topkdf[2] == topkdf[0]) | (topkdf[3] == topkdf[0]), np.int(1), 0) 
        elif k == 4:
            topkdf['top_k_pred'] = np.where((topkdf[1] == topkdf[0]) | (topkdf[2] == topkdf[0]) | (topkdf[3] == topkdf[0]) | (topkdf[4] == topkdf[0]), np.int(1), 0) 
        elif k == 5:
            topkdf['top_k_pred'] = np.where((topkdf[1] == topkdf[0]) | (topkdf[2] == topkdf[0]) | (topkdf[3] == topkdf[0]) | (topkdf[4] == topkdf[0]) | (topkdf[5] == topkdf[0]), np.int(1), 0) 

        #print(topkdf['top_k_pred'])
        topkdf['class_total'] = 1 # all values

        # Get top k accuracy per class
        predictiondf = topkdf.groupby([0]).sum().reset_index()
        predictiondf['per_class_acc'] = predictiondf['top_k_pred']/predictiondf['class_total']
        topkaccdf = pd.concat([pd.Series(animallabels),predictiondf['per_class_acc']], ignore_index=True,axis=1)

        print('\033[1m','Top-',k,' Accuracy per class','\033[0m')
        print('\033[1m', 'Top-',k,' Accuracy = ', np.mean(predictiondf['per_class_acc']),'\033[0m')
        print(topkaccdf, "\n")
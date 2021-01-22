import sys
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import SegmentAnalysis as SA
# Sklearn does not automatically import submodules
import sklearn.ensemble as skle
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import sklearn.discriminant_analysis as sklda
import sklearn.preprocessing as sklp
import sklearn.metrics as sklm

def ModelAccuracy():
    # Set up cmd arguments for the user
    parser = argparse.ArgumentParser(description = 'Enter file names')
    parser.add_argument('filename_gtruth', type = str, help = 'Features extracted from algorithm plus ground truth (one sheet)\n'
        + 'Named "test_gtruth" + date + ".xlsx" from FeatureExploration.py')
    parser.add_argument('predictions', type = str, help = 'Model predictions produced by FeatureSelection.py\n'
        + 'Named "new_images_predictions.xlsx" from FeatureSelection.py')
    args = parser.parse_args()

    # Open excel file
    gtruth = pd.read_excel(args.filename_gtruth).drop(['Unnamed: 0'], axis = 1)

    # Open excel file
    y_pred = pd.read_excel(args.predictions).drop(['Unnamed: 0'], axis = 1)

    # Check results
    combined = pd.merge(y_pred, gtruth, on='Name', how='inner')
    combined.to_excel('model_selection.xlsx')

    # Analyze by strain
    my_strains = combined.groupby(['Folder'])
    for strain in my_strains:
        pass

    # Analyze all strains together
    combined = combined.dropna().drop(['Folder', 'Name', 'Num'], axis = 1)
    o = SA.confusion_matrix(combined['LDAA_Whole_Animal'], combined['Whole_Animal'], 'LDA Whole Animal Check A')
    print('LDA accuracy for whole animal new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['LDAA_Single_Loaded'], combined['Single_Loaded'], 'LDA Single Loaded Check A')
    print('LDA accuracy for single loaded new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['LDAA_Clear'], combined['Clear'], 'LDA Clear Check A')
    print('LDA accuracy for clear new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['LDAA_Straight'], combined['Straight'], 'LDA Straight Check A')
    print('LDA accuracy for straight new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['LDAA_Head_First'], combined['Head_First'], 'LDA Head First Check A')
    print('LDA accuracy for head first new images: ' + str(o[0]))

    o = SA.confusion_matrix(combined['SFSA_Whole_Animal'], combined['Whole_Animal'], 'SFS Whole Animal Check A')
    print('SFS accuracy for whole animal new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['SFSA_Single_Loaded'], combined['Single_Loaded'], 'SFS Single Loaded Check A')
    print('SFS accuracy for single loaded new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['SFSA_Clear'], combined['Clear'], 'SFS Clear Check A')
    print('SFS accuracy for clear new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['SFSA_Straight'], combined['Straight'], 'SFS Straight Check A')
    print('SFS accuracy for straight new images: ' + str(o[0]))
    o = SA.confusion_matrix(combined['SFSA_Head_First'], combined['Head_First'], 'SFS Head First Check A')
    print('SFS accuracy for head first new images: ' + str(o[0]))

    plt.matshow(combined.corr())
    plt.title('Correlation between output and ground truth', pad = 60)
    plt.xticks(range(len(combined.columns)), combined.columns, rotation=90)
    plt.yticks(range(len(combined.columns)), combined.columns)
    plt.savefig('model_selection/Correlation between output and ground truth.png', bbox_inches = "tight")
    plt.show()

    y_true = combined[['Single_Loaded', 'Whole_Animal', 'Straight', 'Clear', 'Head_First']].values.ravel()
    y_pred_lda = combined[['LDAA_Whole_Animal', 'LDAA_Single_Loaded', 'LDAA_Clear', 'LDAA_Straight',
        'LDAA_Head_First']].values.ravel()
    y_pred_sfs = combined[['SFSA_Whole_Animal', 'SFSA_Single_Loaded', 'SFSA_Clear', 'SFSA_Straight',
        'SFSA_Head_First']].values.ravel()
    print('\tTraining accuracy on selected features: %.3f' % sklm.accuracy_score(y_true, y_pred_lda))
    print('\tTraining mean_abs_error on selected features: %.3f' % SA.mean_abs_error(y_true, y_pred_lda))
    print('\tTesting accuracy on selected features: %.3f' % sklm.accuracy_score(y_true, y_pred_sfs))
    print('\tTesting mean_abs_error on selected features: %.3f' % SA.mean_abs_error(y_true, y_pred_sfs))


ModelAccuracy()
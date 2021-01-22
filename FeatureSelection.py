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

def FeatureSelection():
    # Set up cmd arguments for the user
    parser = argparse.ArgumentParser(description = 'Enter file names')
    parser.add_argument('filename_gtruth1', type = str, help = 'Features extracted from algorithm plus ground truth (one sheet)\n'
        + 'Named "train_merged_data_" + date + ".xlsx" from FeatureExploration.py')
    parser.add_argument('filename_new', type = str, help = 'Features extracted from algorithm that do not have a ground truth (one sheet)\n'
        + 'Named "test_" + date + ".xlsx" from FeatureExploration.py')
    args = parser.parse_args()

    # Open excel file
    dt0 = pd.read_excel(args.filename_gtruth1).drop(['Unnamed: 0'], axis = 1)
    main_five = ['Single_Loaded', 'Whole_Animal', 'Straight', 'Clear', 'Head_First']
    # Drop all attributes which are not integers or floats
    dt = pd.concat([dt0.select_dtypes('int64'), dt0.select_dtypes('float64')], axis = 1).drop(['Num'], axis = 1)
    dt = dt.dropna()
    # Set up variables to use for later
    extracted_features = dt.drop(main_five, axis = 1).columns
    gtruth_pure = pd.concat([dt0.copy()[main_five], dt0.copy()['Name']], axis = 1)


    #####################################################################################
    # LASSO: Least Absolute Shrinkage and Selection Operator: does feature
    # selection for you for linear model (L1 Regression)
    #####################################################################################
    # # Lasso
    # print('\nLasso')
    # lass1, grid1 = SA.my_lasso(attributes, dt['Single_Loaded'], 'Single_Loaded')
    # lass2, grid2 = SA.my_lasso(attributes, dt['Clear'], 'Clear')
    # lass3, grid3 = SA.my_lasso(attributes, dt['Straight'], 'Straight')
    # lass4, grid4 = SA.my_lasso(attributes, dt['Head_First'], 'Head_First')
    # lass5, grid5 = SA.my_lasso(attributes, dt['Whole_Animal'], 'Whole_Animal')
    # lass_weights = pd.DataFrame([lass1.coef_, lass2.coef_, lass3.coef_, lass4.coef_, lass5.coef_])
    # lass_weights.columns = attributes.columns
    # lass_weights['Features'] = main_five
    # lass_weights.to_excel('feature_selection/lasso_weights.xlsx')
    # # Plot Lasso Weights
    # plt.matshow(lass_weights.drop(['Features'], axis = 1))
    # plt.xticks(range(len(attributes.columns)), attributes.columns, rotation=90)
    # plt.yticks(range(len(main_five)), main_five)
    # plt.title('Lasso Weights for Each Classification', pad = 120)
    # plt.savefig('feature_selection/lasso_weights.png')
    # plt.show()

    # a = lass_weights.drop(['Features'], axis = 1).T
    # a.columns = main_five
    # for feature in main_five:
    #     print(feature)
    #     print(a.abs().nlargest(5, feature)[feature])


    # L1 regression
    # print('\nL1')
    # l1_1, coefs1 = SA.l1_reg(attributes, dt['Single_Loaded'], 'Single_Loaded')
    # l1_2, coefs2 = SA.l1_reg(attributes, dt['Clear'], 'Clear')
    # l1_3, coefs3 = SA.l1_reg(attributes, dt['Straight'], 'Straight')
    # l1_4, coefs4 = SA.l1_reg(attributes, dt['Head_First'], 'Head_First')
    # l1_5, coefs5 = SA.l1_reg(attributes, dt['Whole_Animal'], 'Whole_Animal')

    # # Heatmap of weights returned from L1
    # for coef, feature in zip([coefs1.T, coefs2.T, coefs3.T, coefs4.T, coefs5.T], main_five):
    #     print(feature)
    #     for i in coef.columns:
    #         print('C = ' + str(i))
    #         print(attributes.columns[coef[i].abs().nlargest(5).index])
    #     plt.matshow(coef)
    #     plt.xticks(range(len(coef.columns)), coef.columns)
    #     plt.yticks(range(len(attributes.columns)), attributes.columns)
    #     plt.title('change in coeffs by c for ' + feature)
    #     plt.savefig('feature_selection/change_in_coeffs_by_c_for_' + feature + '.png')
    #     plt.show()
    #     print('\n')


    #####################################################################################
    # LDA
    #####################################################################################
    # Use lda function from SegmentAnalysis.py
    print('\nLDA with image filtering')
    dt_lda = dt.copy()
    # Run LDA on the first features of the main five
    test_acc2, lda2 = SA.lda(dt_lda.drop(main_five, axis = 1),
        dt_lda['Whole_Animal'], 'Whole_Animal')
    # After running the LDA, remove all rows which had images that were NOT single loaded.
    # Otherwise, this could mess up the data because if the image has multiple worms, the
    # other features (whole animal, straight, clear, and head first) could not have been
    # determined. This is true for each of the following features
    dt_lda = dt_lda[dt_lda['Whole_Animal'] == 1]
    test_acc1, lda1 = SA.lda(dt_lda.drop(main_five, axis = 1),
        dt_lda['Single_Loaded'], 'Single_Loaded')
    dt_lda = dt_lda[dt_lda['Single_Loaded'] == 1]
    test_acc3, lda3 = SA.lda(dt_lda.drop(main_five, axis = 1),
        dt_lda['Straight'], 'Straight')
    dt_lda = dt_lda[dt_lda['Straight'] == 1]
    test_acc4, lda4 = SA.lda(dt_lda.drop(main_five, axis = 1),
        dt_lda['Clear'], 'Clear')
    dt_lda = dt_lda[dt_lda['Clear'] == 1]
    test_acc5, lda5 = SA.lda(dt_lda.drop(main_five, axis = 1),
        dt_lda['Head_First'], 'Head_First')

    # Put all of the weight coefficients from lda into a single pandas data frame
    lda_weights = pd.DataFrame([lda1.coef_[0], lda2.coef_[0], lda3.coef_[0], lda4.coef_[0], lda5.coef_[0]])
    lda_weights.columns = extracted_features
    # Get the absolute values of the weights for later
    abs_lda_weights = lda_weights.abs()
    # Add a column designating which feature corresponds to which row of the data frame
    lda_weights.index = main_five
    # Plot weights
    plt.matshow(lda_weights)
    plt.colorbar()
    plt.yticks(range(len(main_five)), main_five)
    plt.title('Coefficient values for each feature from LDA', pad = 65)
    plt.xticks(range(len(extracted_features)), extracted_features, rotation = 90)
    plt.savefig('feature_selection/lda_weights_heatmap.png', bbox_inches = "tight")
    plt.show()
    # Output the weights to an excel file
    lda_weights.to_excel('feature_selection/lda_weights.xlsx')

    # Identify largest weights
    abs_lda_weights = abs_lda_weights.T
    abs_lda_weights.columns = main_five
    for column in main_five:
        print(column + ': \n')
        print(abs_lda_weights.nlargest(5, column)[column])


    #####################################################################################
    # Forward/backward/stepwise selection: only keep the best/most accurate
    # variables (ML extend module)
    #####################################################################################
    # Step forward feature selection from SegmentAnalysis.py
    print('\nSFS with image filtering')
    dt_sfs = dt.copy()
    # Run SFS on the first feature of the main five
    # After running the SFS, remove all rows which had images that were NOT single loaded.
    # Otherwise, this could mess up the data because if the image has multiple worms, the
    # other features (whole animal, straight, clear, and head first) could not have been
    # determined. This is true for each of the following features
    sfs2, classifier2, data2 = SA.step_forward(dt_sfs.drop(main_five, axis = 1),
        dt_sfs['Whole_Animal'], 'Whole_Animal')
    dt_sfs = dt_sfs[dt_sfs['Whole_Animal'] == 1]
    sfs1, classifier1, data1 = SA.step_forward(dt_sfs.drop(main_five, axis = 1),
        dt_sfs['Single_Loaded'], 'Single_Loaded')
    dt_sfs = dt_sfs[dt_sfs['Single_Loaded'] == 1]
    sfs3, classifier3, data3 = SA.step_forward(dt_sfs.drop(main_five, axis = 1),
        dt_sfs['Straight'], 'Straight')
    dt_sfs = dt_sfs[dt_sfs['Straight'] == 1]
    sfs4, classifier4, data4 = SA.step_forward(dt_sfs.drop(main_five, axis = 1),
        dt_sfs['Clear'], 'Clear')
    dt_sfs = dt_sfs[dt_sfs['Clear'] == 1]
    sfs5, classifier5, data5 = SA.step_forward(dt_sfs.drop(main_five, axis = 1),
        dt_sfs['Head_First'], 'Head_First')

    sfs_data = pd.concat([data1, data2, data3, data4, data5])
    sfs_data.to_excel('feature_selection/sfs_data.xlsx')


    #####################################################################################
    # New Images: Predictions
    #####################################################################################
    # Open excel file
    dt_new = pd.read_excel(args.filename_new)
    # Set up output data frame
    output = pd.concat([dt_new.select_dtypes('object'), dt_new['Num']], axis = 1)
    # Drop all attributes which are not integers or floats
    new_images = pd.concat([dt_new.select_dtypes('int64'), dt_new.select_dtypes('float64')], axis = 1).drop(['Unnamed: 0', 'Num'], axis = 1)
    sc = sklp.StandardScaler()
    new_images_std = sc.fit_transform(new_images)
    output['LDA_Single_Loaded'] = lda1.predict(new_images_std)
    output['LDA_Whole_Animal'] = lda2.predict(new_images_std)
    output['LDA_Straight'] = lda3.predict(new_images_std)
    output['LDA_Clear'] = lda4.predict(new_images_std)
    output['LDA_Head_First'] = lda5.predict(new_images_std)
    output['SFS_Single_Loaded'] = classifier1.predict(new_images_std[0:,list(sfs1.k_feature_idx_)])
    output['SFS_Whole_Animal'] = classifier2.predict(new_images_std[0:,list(sfs2.k_feature_idx_)])
    output['SFS_Straight'] = classifier3.predict(new_images_std[0:,list(sfs3.k_feature_idx_)])
    output['SFS_Clear'] = classifier4.predict(new_images_std[0:,list(sfs4.k_feature_idx_)])
    output['SFS_Head_First'] = classifier5.predict(new_images_std[0:,list(sfs5.k_feature_idx_)])
    # Output
    output.to_excel('new_images_predictions.xlsx')

FeatureSelection()
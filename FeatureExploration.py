# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import SegmentAnalysis as SA
import sklearn.model_selection as sklms
import mlxtend.feature_selection as mlx

# Turn off interactive plotting (uncomment this if you want to show the plots
# as they are created and replace plt.close() with plt.show())
# plt.ioff()

def FeatureExploration():
    # Set up data output
    writer1 = pd.ExcelWriter('feature_exploration/correl_data.xlsx', engine = 'xlsxwriter')

    # Set up cmd arguments for the user
    parser = argparse.ArgumentParser(description = 'Enter file names')
    parser.add_argument('filename', type = str, help = 'Features extracted from algorithm (one sheet)')
    parser.add_argument('gtruth', type = str, help = 'Manual ground truth data (one sheet)')
    args = parser.parse_args()

    # Extract data from csv files
    df = pd.read_csv(args.filename, index_col = False)
    gt = pd.read_excel(args.gtruth, index_col = False)
    main_five = ['Single_Loaded', 'Clear', 'Straight', 'Head_First', 'Whole_Animal']

    # Merge the two data tables based on the images that each file have in common and drop
    # all rows with NaN values
    dt = pd.merge(df, gt, on='Name', how='inner').dropna()
    train, test = sklms.train_test_split(dt.index)
    train = train.values.tolist()
    test = test.values.tolist()
    dt.iloc[train].to_excel('train_merged_data_' + date.today().strftime('%m-%d-%Y') + '.xlsx')
    # Output files for future use
    dt.drop(main_five, axis = 1).iloc[test].dropna().to_excel('test_' + date.today().strftime('%m-%d-%Y')
        + '.xlsx')
    pd.concat([dt.iloc[test][main_five],dt.iloc[test]['Name']], axis = 1).dropna().to_excel('test_gtruth_' + date.today().strftime('%m-%d-%Y')
        + '.xlsx')

    # Set up a data frame with only numbers
    nums_only_dt = pd.concat([dt.select_dtypes('int64'), dt.select_dtypes('float64')], axis = 1)

    # Set the strain to be the index instead of the image name now that we have
    # combined the files
    strains = dt.Folder.unique()
    features = nums_only_dt.drop(main_five, axis = 1).columns

    #####################################################################################
    # Can drop one feature if it is pairwise correlating with another
    # feature (reduce redundancy)
    #####################################################################################
    # Correlation, regarless of strain
    print('Correlation')
    corr = dt.corr()
    # Show figure
    plt.matshow(corr, fignum = 10)
    # Plot
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.rcParams.update({'font.size':10})
    plt.gcf().set_size_inches(14.5, 14.5)
    plt.title('Correlation of All Strains', pad = 60)
    plt.savefig('feature_exploration/corr.png')
    plt.close()

    # Identify all relationships that are more correlated than 0.95
    sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
    corr.to_excel(writer1, sheet_name = 'all_strains')
    sig.to_excel(writer1, sheet_name = 'all_strains_above_0.95')

    # Look at correlation by strain
    for group, strain in zip(dt.groupby(['Folder']), strains):
        corr = group[1].corr()
        plt.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar()
        plt.gcf().set_size_inches(14.5, 14.5)
        plt.title('Correlation between features in strain ' + strain, pad = 60)
        plt.savefig('feature_exploration/' + strain + '_corr.png')
        plt.close()

        corr.to_excel(writer1, sheet_name = strain)
        sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
        sig.to_excel(writer1, sheet_name = strain + '_above_0.95')

    writer1.save()

    #####################################################################################
    # PCA: uses orthogonal transformation to reduce excessive multicollinearity,
    # suitable for unsupervised learning when explanation of predictors
    # is not important
    #####################################################################################
    print('PCA')
    pca, principalDf = SA.my_pca(nums_only_dt, features, 8)
    pca_corr_dt = pd.DataFrame()
    for feature in main_five:
        out = SA.pca_correlation(nums_only_dt[feature], principalDf, feature)
        pca_corr_dt[feature] = out
        SA.pca_plot(nums_only_dt, principalDf, feature)
        highest_corr = out.nlargest(2).index
        # Plot the principal components which are most highly correlated with each main feature
        SA.pca_plot(nums_only_dt, principalDf, feature, highest_corr[0], highest_corr[1])


FeatureExploration()
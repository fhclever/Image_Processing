import matplotlib.pyplot as plt
import mlxtend.feature_selection as mlx
import mlxtend.plotting as mlxp
import numpy as np
import pandas as pd
import seaborn as sns
import sys
# Sklearn does not automatically import submodules
import sklearn.ensemble as skle
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import sklearn.decomposition as skl
import sklearn.preprocessing as sklp
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as sklm
import sklearn.neighbors as skln

# Turn off interactive plotting (uncomment this if you want to show the plots
# as they are created and replace plt.close() with plt.show())
plt.ioff()


'''
LIVE BODY ANALYSIS
'''


def stats(img):
    MeanInt = np.mean(img[np.nonzero(img)])
    stdInt = np.std(img[np.nonzero(img)])
    perct95 = np.percentile(img[np.nonzero(img)],95)
    medianInt = np.percentile(img[np.nonzero(img)],50)
   
    [row, col] = np.nonzero(img)
    avgWidth = max(row) - min(row)

    SumInt = np.sum(img[np.nonzero(img)])

    return [MeanInt, avgWidth, stdInt, perct95, medianInt, SumInt]


'''
FEATURE EXPLORATION
'''


def standardize(df, features):
    scaler = sklp.StandardScaler()
    scaled_dt = pd.DataFrame(scaler.fit_transform(df.values))
    scaled_dt.columns = features
    return scaled_dt

def pca_correlation(original_dt, pca_data, target):
    corr = pca_data.corrwith(original_dt)
    plt.plot(corr)
    plt.xlabel('Correlation between ' + target + ' and extracted features PCA')
    plt.title('Correlation of PCs with original Features: Analyzing ' + target)
    plt.savefig('feature_exploration/pca_corr_' + target + '_n=8.png')
    plt.close()
    return corr

# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
def my_pca(dt, features, n = 2):
    x = dt.loc[:, features]
    scaled_dt = standardize(x, features)
    pca = skl.PCA(n_components = n)

    # principalComponents variable represented the original scaled_dt projected to the desired
    # orthogonal components
    principalComponents = pca.fit_transform(scaled_dt)
    print("Explained Variance Ratio: " + str(pca.explained_variance_ratio_))

    # Eigenvalues
    eigenvalues = pca.explained_variance_

    # Set up pca output data frame
    cols = []
    for i in range(n):
        cols.append('pc' + str(i + 1))
    principalDf = pd.DataFrame(data = principalComponents, columns = cols)

    return pca, principalDf

def pca_plot(dt, principalDf, target, pc1 = 'pc1', pc2 = 'pc2'):
    # Create data from PCA output
    finalDf = pd.concat([principalDf, dt[target]], axis = 1)

    # Graph PCA
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(pc1, fontsize = 15)
    ax.set_ylabel(pc2, fontsize = 15)
    ax.set_title('PCA: ' + target, fontsize = 20)
    colors = ['r', 'g']
    for i, color in zip((0,1), colors):
        indicesToKeep = finalDf[target] == i
        ax.scatter(finalDf.loc[indicesToKeep, pc1], finalDf.loc[indicesToKeep, pc2],
            c = color, s = 50)
    ax.legend(['Not ' + target, target])
    ax.grid()
    fig.savefig('feature_exploration/PCA_' + target + '_' + pc1 + '_' + pc2 + '.png')
    plt.close()


'''
FEATURE SELECTION
'''

# Helper functions
def tts_std(indep_vars, dep_var, test_size = 0.3, random_state = 0):
    # Divide your data into testing and training data to look at how well Lasso does in predicting
    # your main features. Random_state should be zero so that there is no shuffling of your data
    # set and test size is set to the common size of one third of your data.
    X_train, X_test, y_train, y_test = sklms.train_test_split(indep_vars, dep_var, test_size = 0.3, 
        random_state = 0)
    # Create a scaler object
    sc = sklp.StandardScaler()
    # Fit the scaler to the training data and transform
    X_train_std = sc.fit_transform(X_train)
    # Apply the scaler to the test data
    X_test_std = sc.fit_transform(X_test)
    # Ravel
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train_std, X_test_std, y_train, y_test

def confusion_matrix(y_true, y_pred, name):
    data = sklm.confusion_matrix(y_true, y_pred)
    size = y_true.shape[0]
    (tn, fp, fn, tp) = data.ravel()
    accuracy = (tp + tn) / size
    fig, ax = plt.subplots()
    ax.imshow(data)
    ax.set_xlabel('Predicted')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual')
    plt.xticks(range(2), ['P', 'N'])
    plt.yticks(range(2), ['P', 'N'])
    for (i, j), z in np.ndenumerate(data):
        if i == 1:
            i -= 0.25
        else:
            i += 0.25
        ax.text(j, i, '{:0.1f}\n{:0.2f}%'.format(z, z/size*100), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.title('Confusion Matrix for ' + name + '\nAccuracy = ' + str(accuracy), pad = 39)
    plt.savefig('feature_selection/conf_mat_' + name + '.png')
    plt.close()

    # Further analysis: binary classification accruacy metrics based on confusion matrix
    # F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / (1 / precision + 1 / recall)
    print('\t' + name + ' f1 score: ' + str(f1))
    return accuracy, f1

def my_auc(y_true, y_pred, name, features = ['feature']):
    # ROC-AUC
    y_pred = y_pred.T
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(name + ' tpr by fpr ROC curve')
    for i, feature in zip(y_pred, features):
        fpr, tpr, thresholds = sklm.roc_curve(y_true, i)
        roc_score = sklm.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = feature + ' ' + str(roc_score))
    plt.legend(loc  = 'best')
    plt.savefig('feature_selection/roc_auc_' + name + '.png')
    plt.close()

    # # PR-AUC
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(name + ' p by r PR curve')
    # for i, feature in zip(y_pred, features):
    #     p, r, thresholds = sklm.precision_recall_curve(y_true, i)
    #     pr_score = sklm.auc(r, p)
    #     plt.plot(r, p, label = feature + ' ' + str(pr_score))
    # plt.legend(loc  = 'best')
    # plt.savefig('feature_selection/pr_auc_' + name + '.png')
    # plt.close()

    return roc_score

def mean_abs_error(y_true, y_pred):
    # Possible range [0, infinity) -- the higher this value is, the worse the model.
    # For us, if every prediction from a model is incorrect, then the difference between
    # original and predicted values for each data point will be either 1 or -1, so the
    # error will end up being N / N or 1. Therefore, the closer our error is to 1, the worse
    # the model and the closer it is to zero, the better.
    # Note this function gives the exact same result as sklearn.metrics.mean_absolute_error()
    # but I'm keeping mine so I can better remember how it is calculated.
    N = y_true.shape[0]
    error = np.sum(np.absolute(np.subtract(y_true, y_pred))) / N
    return error

# Feature selection classifiers/models
def my_lasso(X, y, name):
    print(name)
    # Set up training/testing standardized data
    X_train_std, X_test_std, y_train, y_test = tts_std(X, y)

    lasso = skllm.Lasso(tol = 1e-1, max_iter = 100000)
    # When alpha = 0, you are just doing linear regression
    parameters = {'alpha':[1e-3, 1e-2, 0.005, 0.0025, 0.0075]}
    # GridSearchCV will give back the best alpha value from parameters. Scoring method is a
    # regression method: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    lasso_regressor = sklms.GridSearchCV(lasso, parameters, scoring = 'neg_mean_absolute_error',
        cv = 5)
    # Fit the data to the GridSearchCV object and the Lasso object
    lasso_regressor.fit(X_train_std, y_train)
    # Ideal alpha value (aka penalty term value)
    print('\t' + str(lasso_regressor.best_params_))
    # Mean cross-validated score of the best_estimator
    print('\t' + str(lasso_regressor.best_score_))

    # Predictions
    lasso_01 = skllm.Lasso(tol = 1e-1, max_iter = 100000, alpha = 0.001)
    lasso_01.fit(X_train_std, y_train)

    # Calling predict on the GridSearchCV object will predict y_train/test_pred based on the estimator
    # with the best found parameters
    y_train_pred = lasso_regressor.predict(X_train_std)
    y_test_pred = lasso_regressor.predict(X_test_std)
    # distplot: plots a histogram of univariate distribution of observations and the estimated pdf
    sns.distplot(y_test - y_test_pred)
    plt.xlabel('Difference between predicted and actual values')
    plt.ylabel('Count')
    plt.title("Lasso: " + name)
    plt.savefig('feature_selection/lasso_' + name + '.png')
    plt.close()

    # ROC-AUC?
    my_auc(y_test, np.array([y_test_pred]).T, 'lasso ' + name)

    # CV scores
    scores = sklms.cross_val_score(lasso_regressor, X, y, cv = 4)
    print('\t' + name + ' CVs: ' + str(scores))

    return lasso_01, lasso_regressor

def l1_reg(X, y, name):
    print(name)
    # Inspiration: https://chrisalbon.com/machine_learning/logistic_regression/logistic_regression_with_l1_regularization/

    # Set up training/testing standardized data
    X_train_std, X_test_std, y_train, y_test = tts_std(X, y)

    # Test multiple C values
    parameters = {'C' : [10, 5, 2, 1.75, 1.5, 1.4, 1.3, 1.2, 1.1, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1]}
    # GridSearchCV will give back the best alpha value from parameters
    clf = skllm.LogisticRegression(penalty='l1', tol = 1e-1, max_iter = 10000, solver='liblinear')
    # Scoring method is a regression method: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    l1_regressor = sklms.GridSearchCV(clf, parameters, scoring = 'neg_mean_absolute_error',
        cv = 5)
    # Fit the data to the GridSearchCV object and the Lasso object
    l1_regressor.fit(X_train_std, y_train)
    # Ideal alpha value (aka penalty term value)
    print('\t' + str(l1_regressor.best_params_))
    # Mean cross-validated score of the best_estimator
    print('\t' + str(l1_regressor.best_score_))

    # Accuracy
    y_train_pred = l1_regressor.predict(X_train_std)
    print('\tTraining accuracy on selected features: %.3f' % sklm.accuracy_score(y_train, y_train_pred))
    print('\tTraining mean absolute error on selected features: %.3f' % mean_abs_error(y_train, y_train_pred))
    y_test_pred = l1_regressor.predict(X_test_std)
    print('\tTesting accuracy on selected features: %.3f' % sklm.accuracy_score(y_test, y_test_pred))
    print('\tTesting mean_abs_error on selected features: %.3f' % mean_abs_error(y_test, y_test_pred))

    # For graphing different C values
    test_accuracy = []
    data = []
    plt.xlabel('Value of C')
    plt.ylabel('Test Accuracy')
    plt.title('L1 Regression Accuracy ' + name)
    plt.xticks(range(len(np.log(parameters['C']))), np.log(parameters['C']), rotation = 90)
    for c in parameters['C']:
        clf = skllm.LogisticRegression(penalty='l1', C=c, solver='liblinear')
        clf.fit(X_train_std, y_train)
        data.append(clf.coef_[0].tolist())
        test_accuracy.append(clf.score(X_test_std, y_test))
    plt.plot(np.log(parameters['C']), test_accuracy)
    plt.savefig('feature_selection/l1_' + name + '.png')
    plt.close()

    # CV scores
    scores = sklms.cross_val_score(l1_regressor, X, y, cv = 4)
    print('\t' + name + ' CVs: ' + str(scores))

    return l1_regressor, pd.DataFrame(data, index = parameters['C'])

def lda(X, y, name):
    print(name)

    # Set up training/testing standardized data
    X_train_std, X_test_std, y_train, y_test = tts_std(X, y)

    # Set up LDA classifier. Default solver is 'svd' or singular value decomposition, which does not compute
    # the covariance matrix and therefore is recommended for data with a large number of features. The param
    # n_components defaults to None and must be less than or equal to either the (number of classes - 1) or
    # the (number of features), whichever one is the lower values. Because we have a binary classification
    # (either clear or blurry, 0 or 1), n_components must be equal to 1.
    clf = sklda.LinearDiscriminantAnalysis(solver = 'svd', n_components = 1)
    clf.fit(X_train_std, y_train)

    # Accuracy
    y_train_pred = clf.predict(X_train_std)
    print('\tTraining accuracy on selected features: %.3f' % sklm.accuracy_score(y_train, y_train_pred))
    print('\tTraining mean absolute error on selected features: %.3f' % mean_abs_error(y_train, y_train_pred))
    y_test_pred = clf.predict(X_test_std)
    print('\tTesting accuracy on selected features: %.3f' % sklm.accuracy_score(y_test, y_test_pred))
    print('\tTesting mean_abs_error on selected features: %.3f' % mean_abs_error(y_test, y_test_pred))

    # Confusion matrix generation
    confusion_matrix(y_train, y_train_pred, name + "_LDA_Training_Data")
    confusion_matrix(y_test, y_test_pred, name + "_LDA_Testing_Data")

    # Plot LD1
    bins = np.linspace(-10, 10, 100)
    # Testing data
    graphing = pd.DataFrame(clf.fit_transform(X_test_std, y_test), y_test)
    present = graphing[graphing.index == 1]
    absent = graphing[graphing.index == 0]
    plt.hist(present.values, bins = bins, alpha = 0.7, rwidth=0.85, label = name + ' testing')
    plt.hist(absent.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Not ' + name + ' testing')
    plt.xlabel("LD1")
    plt.ylabel('Count')
    plt.legend(loc = 'best')
    plt.title('LDA projections for ' + name + ' training data')
    plt.savefig('feature_selection/lda_training_' + name + '.png')
    plt.close()
    # Training data
    graphing = pd.DataFrame(clf.fit_transform(X_train_std, y_train), y_train)
    present = graphing[graphing.index == 1]
    absent = graphing[graphing.index == 0]
    plt.hist(present.values, bins = bins, alpha = 0.7, rwidth=0.85, label = name + ' training')
    plt.hist(absent.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Not ' + name + 'training')
    plt.xlabel("LD1")
    plt.ylabel('Count')
    plt.legend(loc = 'best')
    plt.title('LDA projections for ' + name + ' testing data')
    plt.savefig('feature_selection/lda_testing_' + name + '.png')
    plt.close()

    # CV scores
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'balanced_accuracy')
    print('\t' + name + ' CVs, scoring = balanced_accuracy: ' + str(scores))
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'f1')
    print('\t' + name + ' CVs, scoring = f1: ' + str(scores))
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'roc_auc')
    print('\t' + name + ' CVs, scoring = roc_auc: ' + str(scores))

    return sklm.accuracy_score(y_test, y_test_pred), clf

def step_forward(X, y, name):
    print(name)
    # Inspiration: https://www.kdnuggets.com/2018/06/step-forward-feature-selection-python.html

    # Set up training/testing standardized data
    X_train_std, X_test_std, y_train, y_test = tts_std(X, y)

    # Build RF classifier to use in feature selection: liblinear solver recommended when you have
    # high dimension dataset, but once you standardize your data, the accuracy of all solvers is
    # pretty much the same. max_iter (maximum number of iterations taken for the solvers to converge.)
    # is set to a higher number than default (100) so that the model will actually converge (lower
    # values cause a no convergence warning).
    clf = skllm.LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter = 100)

    # Build step forward feature selection: cv (cross validation) is set to zero for no
    # cross validation, k_features = 3 means we are selecting the 3 best attributes to desribe
    # our feature, and verbose is just used for logging the progress of the feature selector
    sfs1 = mlx.SequentialFeatureSelector(clf, k_features=5, forward=True, floating=False, verbose=0,
        scoring='accuracy', cv=10)

    # Perform SFS
    sfs1 = sfs1.fit(X_train_std, y_train, custom_feature_names = X.columns)

    # Which features?
    print('\t' + 'Top 5 features: ' + str(sfs1.k_feature_names_))
    feat_cols1 = list(sfs1.k_feature_idx_)

    # Build full model with selected features: sfs has no predict function
    clf = skllm.LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter = 100)
    # Now that we have the relevant features according to SFS, we can use logistic regression
    # on JUST those features and see how accurately they can predict the classification of
    # single loaded, clear, straight, etc.
    clf.fit(X_train_std[:, feat_cols1], y_train)

    # 'kind' represents the kind of error bar you get in your plot {'std_dev', 'std_err', 'ci',
    # None}. This error bar is the error of the cv scores.
    fig1 = mlxp.plot_sequential_feature_selection(sfs1.get_metric_dict(), kind = 'std_dev')
    plt.title('Sequential Forward Feature Selection CV Scores: ' + name + ' (std dev)')
    plt.ylabel('Mean CV Score')
    plt.grid()
    plt.savefig('feature_selection/sfs_' + name + ".png")
    plt.close()

    # Accuracy
    y_train_pred = clf.predict(X_train_std[:, feat_cols1])
    print('\tTraining accuracy on selected features: %.3f' % sklm.accuracy_score(y_train, y_train_pred))
    print('\tTraining mean absolute error on selected features: %.3f' % mean_abs_error(y_train, y_train_pred))
    y_test_pred = clf.predict(X_test_std[:, feat_cols1])
    print('\tTesting accuracy on selected features: %.3f' % sklm.accuracy_score(y_test, y_test_pred))
    print('\tTesting mean_abs_error on selected features: %.3f' % mean_abs_error(y_test, y_test_pred))

    # Confusion matrix generation
    confusion_matrix(y_train, y_train_pred, name + "_sfs_Training_Data_")
    confusion_matrix(y_test, y_test_pred, name + "_sfs_Testing_Data_")
    my_auc(y_train, X_train_std[:, feat_cols1], name + '_sfs_training', sfs1.k_feature_names_)

    # CV scores
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'balanced_accuracy')
    print('\t' + name + ' CVs, scoring = balanced_accuracy: ' + str(scores))
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'f1')
    print('\t' + name + ' CVs, scoring = f1: ' + str(scores))
    scores = sklms.cross_val_score(clf, X = X_train_std, y = y_train, cv = 5, scoring = 'roc_auc')
    print('\t' + name + ' CVs, scoring = roc_auc: ' + str(scores))

    return sfs1, clf, pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
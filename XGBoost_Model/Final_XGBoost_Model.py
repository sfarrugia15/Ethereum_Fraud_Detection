import time
from collections import Counter

import numpy
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, \
    precision_recall_curve
import xgboost as xgb
import shap
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
import numpy as np

from Account_Stats.Generate_Fields import account_balance


def XGBoost_Classifier (X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(max_depth=4,
                              subsample=0.5,
                              objective='binary:logistic',
                              n_estimators=300,
                              learning_rate=0.2)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric=["error", "logloss"],
              eval_set=eval_set, verbose=True)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    plot_xgb_feature_importance(model, 2, 'weight', width=200, height=200)

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    plt.rcParams.update({'font.size': 13})
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.xlabel('Number of iterations')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.xlabel('Number of iterations')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()

def XGBoost(X_train, y_train, X_test, y_test):
    model2 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.2, subsample=0.5, objective='binary:logistic')

    X_train = X_train.loc[:, X_train.columns != 'Address']
    X_test_addresses = X_test['Address']
    X_test = X_test.loc[:, X_test.columns != 'Address']
    X_test_addresses = np.array(X_test_addresses)

    #train_model1 = model1.fit(X_train, y_train)
    train_model2 = model2.fit(X_train, y_train)
    pred2 = train_model2.predict(X_test)
    #print(classification_report(y_test, pred1))
    #print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))

    print(classification_report(y_test, pred2))
    print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
    auc = roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1])
    precision, recall, threshold = precision_recall_curve(y_test, model2.predict_proba(X_test)[:, 1])
    #print(threshold)
    print("ROC: " , auc)
    #print([i for i, j in zip(pred2, y_test) if i != j])
    index = 0
    FP = []
    FN = []
    for i,j in zip(pred2, y_test):
        if j == 1:
            if i != j:
                FN.append(X_test_addresses[index])

        if j == 0:
            if i != j:
                FP.append(X_test_addresses[index])
        index += 1

    print("FALSE POSITIVES: ", len(FP))
    print("FALSE NEGATIVES: ", len(FN))

    #plot_FP_FN(FP=FP, FN=FN)

    # Weight - number of times feature is used to split the data across all trees
    # Cover - number of times feature is used to split data across all trees weighted by number of
    #         training data points that go through those splits
    # Gain - average training loss reduction gained when using a feature for splitting

    importance = model2.get_booster().get_score(importance_type='weight')
    importance = sorted(importance.items(), key= lambda l:l[1], reverse=True)

    # plot_tree(train_model2)
    # fig = plt.gcf()
    # fig.set_size_inches(150, 100)
    # fig.savefig('tree.png')
    # results = cross_show()val_score(model2, X, Y)
    np.set_printoptions(precision=2)

    # explainer = shap.TreeExplainer(train_model2)
    # shap_values = explainer.shap_values(X_train)
    # print(np.shape(shap_values))
    #shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])

    #shap.force_plot(explainer.expected_value, shap_values, X_train)
    # shap.summary_plot(shap_values, X_train, max_display=10,show=False)
    # fig = plt.gcf()
    # fig.set_size_inches(20,20)
    # fig.savefig('test.png')
    # #shap.summary_plot(shap_values, X_train, max_display=10, plot_type="bar")

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(y_test, pred2, classes=np.array(['Normal','Illicit']),
    #                     title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    # plt.show()
    return importance

def plot_FP_FN(FP,FN):
    import plotly.plotly as py
    import plotly.graph_objs as go
    py.sign_in('sfarr15', 'W10VRtI3WkpBD8gJmP4R')
    trace = go.Table(
        header=dict(values=['False Positives','False Negatives'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    align=['left'] * 5),
        cells=dict(values=[FP,
                           FN],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   align=['left'] * 5))

    layout = dict(width=1000, height=900)
    data = [trace]
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='styled_table')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    print(tn, fp, fn, tp)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax





def stratified_k_fold_XGBoost(X, Y, n_folds):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    print("Starting stratified Cross-validation using XG-Boost")
    #model1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
    #                           subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
    #                           scale_pos_weight=1, seed=27)
    model1 = xgb.XGBClassifier(learning_rate=0.2)
    max_depth = [2, 3, 4, 5, 6, 7, 8]
    n_estimators = [100, 150, 200, 250, 300]
    # max_depth = [ 3, 4 ]
    # n_estimators = [ 150, 200]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    #scoring = {'auc': 'roc_auc', 'acc': 'accuracy'}
    score = {'f1', 'roc_auc', 'accuracy'}
    current = time.time()
    #results = cross_validate(model1, X, Y, cv=kfold, scoring=scoring, return_train_score=True, n_jobs=-1)
    results = GridSearchCV(model1,param_grid, cv=kfold, scoring='roc_auc', n_jobs=6)
    grid_result = results.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Execution time: ", time.time() - current)

    # plot results
    scores = numpy.array(means).reshape(len(max_depth), len(n_estimators))
    for i, value in enumerate(max_depth):
        pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
    pyplot.legend()
    pyplot.xlabel('n_estimators')
    pyplot.ylabel('AUC')
    pyplot.savefig('n_estimators_vs_max_depth.png')
    # print("Test AUC: %.3f STD(%.3f)"% (np.mean(results['test_auc']), np.std(results['test_auc'])),
    #       " Test Accuracy: %.3f STD(%.3f) " % (np.mean(results['test_acc']), np.std(results['test_acc'])))
    # print("--------------------------------------------------------------------------------------------\n")
    # print("Model execution time over %s folds: %s seconds" % (n_folds, time.time()-current))



def random_forest(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc_model = rfc.fit(X_train, y_train)
    pred8 = rfc_model.predict(X_test)
    print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))
    print(classification_report(y_test, pred8))
    roc = roc_auc_score(y_test, rfc_model.predict_proba(X_test)[:, 1])
    print("ROC:" , roc)


def k_means(X, Y):
    addresses = np.array(X['Address'])
    balances = []
    #num_of_transactions = np.array(X)
    num_of_erc20_transactions = np.array(X[" Total ERC20 tnxs"])

    for address in addresses[:100]:
        balances.append(account_balance(address))

    z= np.array(list(zip(balances, num_of_erc20_transactions)))
    LABEL_COLOR_MAP = {0: 'r',
                       1: 'k'}
    kmeans = KMeans(n_clusters=2, random_state=0).fit(z)
    label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
    plt.scatter(z[:,0], z[:,1], c=label_color)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('Balances')
    plt.ylabel('Number of ERC20 transactions')
    plt.xlim((0, 0.001))
    plt.show()

    print(kmeans.labels_)

def prepare_dataset_split(X, Y, testSize):
    test_size = testSize

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, shuffle=True)
    print("Size of Training: ", len(X_train))
    print("Size of Testing: ", len(X_test))
    return X_train, X_test, y_train, y_test

def get_dataset(filename):
    import pandas as pd

    csv_file = filename
    df = pd.read_csv(csv_file)

    Y = df['FLAG']
    X = df.loc[:, df.columns != 'FLAG']
    X.pop('Index')
    X.pop('ERC20_most_sent_token_type')
    X.pop('ERC20_most_rec_token_type')
    X.pop('ERC20_uniq_sent_token_name')
    X.pop('ERC20_uniq_rec_token_name')
    #X.pop('Address')

    X.fillna(0, inplace=True)
    return X, Y

def plot_xgb_feature_importance(model, model_number, importance_type, width, height):
    plt.figure(model_number)
    #plt.rcParams.update({'font.size': 15})
    xgb.plot_importance(model, importance_type=importance_type, show_values=False,
                        xlabel="Frequency") #max_num_features=10
    plt.rcParams['figure.figsize'] = [width, height]
    plt.tight_layout()
    plt.title(importance_type)
    plt.savefig(" "+importance_type)
    plt.show()

# Returns the total frequency per feature over the number of folds
def update_list(importance_value_list):
    c = Counter()
    for k,v in importance_value_list:
        c[k] += v

    # PLOT TOP 10
    # feature_list = list(c.items())[:10]

    # PLOT ALL
    feature_list = list(c.items())

    sorted_feature_list = sorted(feature_list.__iter__(), key=lambda x : x[1], reverse=False)
    print(sorted_feature_list)

    return sorted_feature_list


def plot_average_importance_values(sorted_feature_list, num_of_train_test):

    key, value = zip(*sorted_feature_list.__iter__())
    value = [int(x / num_of_train_test) for x in value]
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20, 10))
    plt.barh(key, value)
    plt.title('Feature importance ranking - Weight')
    plt.xlabel('Frequency')
    plt.ylabel('Features')
    plt.show()

if __name__ == '__main__':
    X, Y = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete.csv')

    #k_means(X,Y)
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/new_illicit_addresses.csv')
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete_Illicit_Subset_1000.csv')
    # stratified_k_fold_XGBoost(X, Y, n_folds=10)

    importance_list = []
    num_of_train_test_splits = 10
    for i in range(num_of_train_test_splits):
        X_train, X_test, y_train, y_test = prepare_dataset_split(X, Y, testSize=0.1)
        #XGBoost(X_train, y_train, X_diff, Y_diff) # TO TEST NEWLY ADDED ADDRESSES
        #XGBoost(X_train, y_train, X_test, y_test)
        importance = XGBoost(X_train, y_train, X_test, y_test)
        importance_list.extend(importance)

    sorted_feature_list = update_list(importance_list)
    #plot_average_importance_values(sorted_feature_list, num_of_train_test_splits)

    #   random_forest(X_train,  y_train, X_test, y_test)



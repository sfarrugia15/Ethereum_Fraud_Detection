from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
def XGBoost(X_train, y_train, X_test, y_test):

    model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.2, subsample=0.5)
    #model2 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
    #                           subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
    #                           scale_pos_weight=1, seed=27)

    #train_model1 = model1.fit(X_train, y_train)
    train_model2 = model2.fit(X_train, y_train)
    #pred1 = train_model1.predict(X_test)
    pred2 = train_model2.predict(X_test)
    #print(classification_report(y_test, pred1))
    #print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))

    print(classification_report(y_test, pred2))
    print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))

    # Weight - number of times feature is used to split the data across all trees
    # Cover - number of times feature is used to split data across all trees weighted by number of
    #         training data points that go through those splits
    # Gain - average training loss reduction gained when using a feature for splitting

    importance = model2.get_booster().get_score(importance_type='weight')
    importance = sorted(importance.items())
    print(importance)

    #plot_xgb_feature_importance(train_model2, 1, 'weight', 20, 10)
    #plot_xgb_feature_importance(train_model2, 2, 'cover', 20, 10)
    #plot_xgb_feature_importance(train_model2, 3, 'gain', 20, 10)

    # results = cross_val_score(model2, X, Y)



def stratified_k_fold_XGBoost(X, Y):
    print("Starting stratified Cross-validation using XG-Boost")
    #model1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
    #                           subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
    #                           scale_pos_weight=1, seed=27)
    model1 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.2, subsample=0.5)
    kfold = StratifiedKFold(n_splits=5)
    scoring = {'auc': 'roc_auc', 'acc': 'accuracy'}
    results = cross_validate(model1, X, Y, cv=kfold, scoring=scoring, return_train_score=True)

    #print(results.keys())
    print("Test AUC: %.3f STD(%.3f)"% (np.mean(results['test_auc']), np.std(results['test_auc'])),
          " Test Accuracy: %.3f STD(%.3f) " % (np.mean(results['test_acc']), np.std(results['test_acc'])))
    print("--------------------------------------------------------------------------------------------\n\n")


def random_forest(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    rfc_model = rfc.fit(X_train, y_train)
    pred8 = rfc_model.predict(X_test)
    print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))


def prepare_dataset_split(X, Y):
    test_size = 0.2

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
    X.pop('Address')
    X.pop(' ERC20 most sent token type')
    X.pop(' ERC20_most_rec_token_type')
    X.pop(' ERC20 uniq sent token name')
    X.pop(' ERC20 uniq rec token name')

    #CHECK IF MODEL IS WORKING BY REMOVING MOST IMPORTANT FEATURES
    # X.pop('total ether balance')
    # X.pop('Time Diff between first and last (Mins)')
    # X.pop('min val sent')
    # X.pop('Avg min between sent tnx')
    # X.pop('max val sent')
    # X.pop('Avg min between received tnx')
    # X.pop('avg val received')
    # X.pop('avg val sent')
    # X.pop('Sent tnx')
    # X.pop('Unique Received From Addresses')
    # X.pop('Unique Sent To Addresses')
    # X.pop('min value received')
    # X.pop('total Ether sent')
    # X.pop('max value received ')
    # X.pop('Received Tnx')
    # X.pop('total ether received')
    # X.pop('total transactions (including tnx to create contract')
    # X.pop(' ERC20 total Ether received')

    X.fillna(0, inplace=True)
    return X, Y

def plot_xgb_feature_importance(model, model_number, importance_type, width, height):
    plt.figure(model_number)
    xgb.plot_importance(model, importance_type=importance_type) #max_num_features=10
    plt.rcParams['figure.figsize'] = [width, height]
    plt.tight_layout()
    plt.title(importance_type)
    plt.savefig(" "+importance_type)
    plt.show()


if __name__ == '__main__':
    X, Y = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete_large.csv')
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/new_illicit_addresses.csv')
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete_Illicit_Subset_1000.csv')
    stratified_k_fold_XGBoost(X, Y)
    for i in range(5):
        X_train, X_test, y_train, y_test = prepare_dataset_split(X, Y)
        #XGBoost(X_train, y_train, X_diff, Y_diff) # TO TEST NEWLY ADDED ADDRESSES
        XGBoost(X_train, y_train, X_test, y_test)
    #random_forest(X_train,  y_train, X_test, y_test)

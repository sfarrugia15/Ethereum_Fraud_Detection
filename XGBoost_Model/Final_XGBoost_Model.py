from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt


def XGBoost(X, Y):
    seed = 7
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)

    model1 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.2, subsample=0.5)

    model2 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0,
                               subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                               scale_pos_weight=1, seed=27)

    train_model1 = model1.fit(X_train, y_train)
    train_model2 = model2.fit(X_train, y_train)
    pred1 = train_model1.predict(X_test)
    pred2 = train_model2.predict(X_test)
    print(classification_report(y_test, pred1))
    print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))

    print(classification_report(y_test, pred2))
    print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))

    # Weight - number of times feature is used to split the data across all trees
    # Cover - number of times feature is used to split data across all trees weighted by number of
    #         training data points that go through those splits
    # Gain - average training loss reduction gained when using a feature for splitting

    plot_xgb_feature_importance(train_model1, 1, 'weight', 20, 10)
    plot_xgb_feature_importance(train_model1, 2, 'cover', 20, 10)
    plot_xgb_feature_importance(train_model1, 3, 'gain', 20, 10)

    # plot_xgb_feature_importance(train_model2, 4, 'weight', 20, 10)
    # plot_xgb_feature_importance(train_model2, 5, 'cover', 20, 10)
    # plot_xgb_feature_importance(train_model2, 6, 'gain', 20, 10)


def get_normal_account_addresses():
    import pandas as pd

    csv_file = 'C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/test.csv'
    df = pd.read_csv(csv_file)

    Y = df['FLAG']
    X = df.loc[:, df.columns != 'FLAG']
    X.pop('Index')
    X.pop('Address')
    X.pop(' ERC20 most sent token type')
    X.pop(' ERC20_most_rec_token_type')
    X.pop(' ERC20 uniq sent token name')
    X.pop(' ERC20 uniq rec token name')

    X.fillna(0, inplace=True)
    return X, Y


def plot_xgb_feature_importance(model, model_number, importance_type, width, height):
    plt.figure(model_number)
    xgb.plot_importance(model, importance_type=importance_type)
    plt.rcParams['figure.figsize'] = [width, height]
    plt.tight_layout()
    plt.title(importance_type)
    plt.savefig(" "+importance_type)
    plt.show()

if __name__ == '__main__':
    X, Y = get_normal_account_addresses()
    XGBoost(X,Y)
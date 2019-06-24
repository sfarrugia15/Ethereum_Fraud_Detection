import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import io, math, ui

NORMAL_FLAG = 0
ILLICIT_FLAG = 1

def read_file(filename):
    import pandas as pd

    csv_file = filename
    df = pd.read_csv(csv_file)

    df = remove_fields(df)

    df.fillna(0, inplace=True)
    return df

# Returns a pandas dataframe file
def read_file_w_flag(filename, flag):
    csv_file = filename
    df = pd.read_csv(csv_file)

    if flag == NORMAL_FLAG:
        normal_accounts = df['FLAG'] == NORMAL_FLAG
        df = df[normal_accounts]

    elif flag == ILLICIT_FLAG:
        illicit_accounts = df['FLAG'] == ILLICIT_FLAG
        df = df[illicit_accounts]
    # Y = df['FLAG']
    # X = df.loc[:, df.columns != 'FLAG']
    df = remove_fields(df)
    df.fillna(0, inplace=True)
    return df


# Remove unnecessary fields
def remove_fields(df):
    df.pop('Index')
    df.pop('Address')
    df.pop(' ERC20 most sent token type')
    df.pop(' ERC20_most_rec_token_type')
    df.pop(' ERC20 uniq sent token name')
    df.pop(' ERC20 uniq rec token name')
    return df

def PCA_plot(df, no_of_components):
    X = df.loc[:, df.columns != 'FLAG']
    Y = df['FLAG']
    print(X.shape)
    pca = PCA(n_components=no_of_components)
    pca_result = pca.fit_transform(X)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    if no_of_components > 3:
        print('Cumulative explained variation for 10 principal components: {}'.format(
            np.sum(pca.explained_variance_ratio_)))

    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x=df["pca-one"], y=df["pca-two"],
    #     hue=Y,
    #     palette=sns.color_palette("hls", 2),
    #     data=X,
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()

def TSNE_plot(df):
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    X = df.loc[:, df.columns != 'FLAG']
    Y = df['FLAG']
    tsne_results = tsne.fit_transform(X)
    X['first_dimension'] = tsne_results[:, 0]
    X['second_dimension'] = tsne_results[:, 1]
    X['third_dimension'] = tsne_results[:,2]

    fig1 = plt.figure(figsize=(16, 10))
    #sns.set(font_scale=1.8)
    sns.scatterplot(
        x="first_dimension", y="second_dimension",
        hue=Y,
        palette=sns.color_palette("hls", 2),
        data=X,
        legend="full",
        alpha=0.3
    )
    fig1.show()
    flag = ["Illicit", "Normal"]
    # defining an array of colors
    colors = ['#2300A8', '#00A658']
    #sns.set(font_scale=1.0)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2,2,1, projection='3d')
    ax.scatter(X["first_dimension"], X["second_dimension"], X["third_dimension"], c=Y, s=3, label=Y)
    ax.view_init(30, 45)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(X["first_dimension"], X["second_dimension"], X["third_dimension"], c=Y, s=3, label=Y)
    ax.view_init(30, 135)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(X["first_dimension"], X["second_dimension"], X["third_dimension"], c=Y, s=3, label=Y)
    ax.view_init(30, 225)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X["first_dimension"], X["second_dimension"], X["third_dimension"], c=Y, s=3, label=Y)
    ax.view_init(30, 315)
    fig.legend(Y, ("illicit", "normal"), "upper-right")
    plt.show()

def k_means(df, K_neighbors):
    Y = df['FLAG']
    X = df.loc[:, df.columns != 'FLAG']
    X = X.loc[:, :]
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    X['tsne-3d-one'] = tsne_results[:, 0]
    X['tsne-3d-two'] = tsne_results[:, 1]
    X['tsne-3d-three'] = tsne_results[:,2]
    classifier = KMeans(n_clusters=K_neighbors)
    classifier.fit(tsne_results[:,:3])
    y_kmeans = classifier.predict(tsne_results[:,:3])

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2,2,1, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'],X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    centers = classifier.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 45)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 135)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 225)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X['tsne-3d-one'], X['tsne-3d-two'], X['tsne-3d-three'], c=y_kmeans, s=20, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    ax.view_init(30, 315)
    plt.show()

if __name__ == '__main__':
    accounts = read_file("C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete.csv")
    TSNE_plot(accounts)

    # PCA_plot(accounts, no_of_components=3)
    # PCA_plot(accounts, no_of_components=10)
    #TSNE_plot(accounts)

    # illicit_accounts = read_file_w_flag("C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete.csv", 1)
    # k_means(accounts, 5)

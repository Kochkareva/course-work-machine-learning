import os.path

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, Birch
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dropout

matplotlib.use('Agg')  # Использование фонового режима рендеринга
import matplotlib.pyplot as plt
import numpy as np

picfld = os.path.join('static', 'charts')


def k_means(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    kmeans = KMeans(n_clusters=9, n_init=10)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    silhouette = silhouette_score(X_test, kmeans.predict(X_test))
    make_plots(X_train, 9, labels, centroids, file_plot_name, silhouette)
    joblib.dump(kmeans, file_model_name)


def birch(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    birch = Birch(n_clusters=9, threshold=0.5, branching_factor=18)
    birch.fit(X_train)
    labels = birch.labels_
    centroids = birch.subcluster_centers_
    silhouette = silhouette_score(X_test, birch.predict(X_test))
    make_plots(X_train, 9, labels, centroids, file_plot_name, silhouette)
    joblib.dump(birch, file_model_name)


def make_plots(x, n_clusters, labels, centroids, file_name, silhouette):
    # Визуализация результатов
    cluster_counts = np.zeros(n_clusters)
    for label in labels:
        cluster_counts[label] += 1
    # Построение гистограммы
    plt.bar(range(n_clusters), cluster_counts, color='#293844')
    # Настройка осей и подписей
    plt.xlabel('Кластер')
    plt.ylabel('Количество элементов')
    plt.title('Силуэтный коэффициент: {0}'.format(silhouette))
    plt.savefig(file_name)
    plt.close()


def get_birch():
    file_plot_name = 'static/charts/birch.png'
    file_model_name = 'static/fit_models/birch.pkl'
    if os.path.isfile(file_plot_name) and os.path.isfile(file_model_name):
        return file_plot_name
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        y = data['Country']
        df.drop(['Country'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, train_size=0.01, test_size=0.0002)
        birch(X_train, X_test, y_train, y_test, file_plot_name, file_model_name)
        return file_plot_name


# оценка количества кластеров
def selection_number_clusters(X_train, X_test, y_train, y_test, file_name_elbow_method):
    inertias = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, n_init=10).fit(X_train, y_train)
        inertias.append(np.sqrt(kmeans.inertia_))
    plt.plot(range(1, 15), inertias, marker='o', color='#293844')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title("Метод локтя")
    plt.savefig(file_name_elbow_method)
    plt.close()


def get_k_means():
    file_plot_name = 'static/charts/k_means.png'
    file_model_name = 'static/fit_models/k_means.pkl'
    file_name_elbow_method = 'static/charts/ElbowMethod.png'
    if os.path.isfile(file_plot_name) and os.path.isfile(file_model_name) and os.path.isfile(file_name_elbow_method):
        return file_name_elbow_method, file_plot_name
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        y = data['Country']
        df.drop(['Country'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, train_size=0.01, test_size=0.0002)
        selection_number_clusters(X_train, X_test, y_train, y_test, file_name_elbow_method)
        k_means(X_train, X_test, y_train, y_test, file_plot_name, file_model_name)
        return file_name_elbow_method, file_plot_name


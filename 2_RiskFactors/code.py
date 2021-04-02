import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None

def read_data():
    # Read data
    data = pd.read_csv("2_RiskFactors/data/riskFactorData.csv", sep=";")
    return data

def get_statistics(data):
    # Get data statistics
    print("\n")
    print(data[["Year", "BMI", "SBP", "FAT"]].describe())
    print("\n")
    print(data[["Country", "Sex"]].describe())
    print("\n")
    print(data[["BMI", "SBP", "FAT"]].corr())

def evaluate(y_true, y_pred):

    print(confusion_matrix(y_true, y_pred))


def knnModel(data):
    # Define year
    year = 2008
    
    # Filter data
    data = data[(data["Year"] == year)]

    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["Sex"].to_list()

    # Define test data
    test_BMI = 23.5
    test_SBP = 126.5
    test_FAT = 4.4
    test_data = [test_BMI, test_SBP, test_FAT]

    for k in range(2, 10):
        # Create kNN classifier
        model = KNeighborsClassifier(n_neighbors=k)
        # Train kNN classifier
        model.fit(x_train, y_train)
        # Predict new sample
        prediction = model.predict([test_data])[0]
        prediction_prob = model.predict_proba([test_data])[0]

        print("\n")
        print(str(k) + "-Nearest Neighbors")
        print("Female: " + str(prediction_prob[0]))
        print("Male: " + str(prediction_prob[1]))
        print("Model Prediction: " + prediction)

def svmModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["AtRisk"].to_list()

    # Define test data
    test_BMI = 29.5
    test_SBP = 126.5
    test_FAT = 4.4
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create SVM classifier
    model = svm.SVC(kernel="rbf", gamma=0.3)
    # Train SVM classifier
    model.fit(x_train, y_train)
    
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)
    
    x_train["pred"] = model.predict(x_train)
    evaluate(data["AtRisk"].to_list(), x_train["pred"].to_list())

def gaussianModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["AtRisk"].to_list()

    # Define test data
    test_BMI = 26.5
    test_SBP = 136.5
    test_FAT = 5.8
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create Gaussian classifier
    model = GaussianNB()
    # Train Gaussian classifier
    model.fit(x_train, y_train)
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)

def decisionTreesModel(data):
    # Keep selected columns
    x_train = data[["BMI", "SBP", "FAT"]]
    y_train = data["AtRisk"].to_list()

    # Define test data
    test_BMI = 25.5
    test_SBP = 136.5
    test_FAT = 4.8
    test_data = [test_BMI, test_SBP, test_FAT]

    # Create Decision Tree classifier
    model = tree.DecisionTreeClassifier()
    # Train Decision Tree classifier
    model.fit(x_train, y_train)
    # Predict new sample
    prediction = model.predict([test_data])[0]
    print("Model Prediction: " + prediction)

def show_clusters(data_frame, labels, x_label, y_label, title):

    plt.scatter(data_frame[x_label], data_frame[y_label], s=3, c = [val + 1 for val in labels])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def compute_silhouette(data_frame, labels):

    return silhouette_score(data_frame, labels)

def perform_kmeans(data_frame, number_of_clusters):

    print("\n --- Perform KMeans ---")
    kmeans_model = KMeans(n_clusters=number_of_clusters, random_state=0).fit(data_frame[["BMI", "FAT"]])

    print("Silhouette:", compute_silhouette(data_frame[["BMI", "FAT"]], kmeans_model.labels_))

    return kmeans_model

def perform_dbscan(data_frame, eps, min_samples):

    print("\n --- Perform DBSCAN ---")
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(data_frame[["BMI", "FAT"]])

    n_clusters_ = len(set(dbscan_model.labels_)) - (1 if -1 in dbscan_model.labels_ else 0)
    print("Number of clusters:", n_clusters_)
    if(n_clusters_ > 1):
        print("Silhouette:", compute_silhouette(data_frame[["BMI", "FAT"]], dbscan_model.labels_))

    return dbscan_model


# Read data
print("Reading data...")
data = read_data()
print("Reading ended...")

# Get general statistics
# get_statistics(data)

# ### CLASSIFICATION ###

# Get sex by kNN
# knnModel(data)

# Get country by SVM
# svmModel(data)

# Get risk by NaiveBayes
# gaussianModel(data)

# Get risk by DecisionTrees
# decisionTreesModel(data)


### CLUSTERING ###

# KMeans Clustering

# filtered_data = data[data["Sex"] == "male"]
# model = perform_kmeans(filtered_data, 2)
# show_clusters(filtered_data, model.labels_, "BMI", "FAT", "KMeans")


# DBSCAN Clustering

# filtered_data = data[data["Sex"] == "male"]
# model = perform_dbscan(filtered_data, 0.12, 20)
# show_clusters(filtered_data, model.labels_, "BMI", "FAT", "DBSCAN")

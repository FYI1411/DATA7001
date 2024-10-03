import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm
from preprocessing_checks import prepare_df

"""
two methods clustering, pca_clustering return 4 and 5 clusters to be optimal respectively.

All we need is compare the clusters returned by both of them and see how they diiffer in terms of all the variables - e.g., 
does Cluster 1 have a higher average GDP than the other clusters? for every cluster.

Report the statistically significant ones

Download wdi.csv and preprocessing_checks.py, put it in directory

"""

def clustering():
    df = prepare_df("wdi.csv")
    df = df[df["Year"] == 2015] # isolate to 2015
    countries = df.loc[:, "Country Name"]
    # covariates = ['Trade (% of GDP)', 'External balance on goods and services (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)', 'Employment in agriculture (% of total employment) (modeled ILO estimate)', 'Employment in industry (% of total employment) (modeled ILO estimate)','Employment in services (% of total employment) (modeled ILO estimate)']
    covariates = ['Trade (% of GDP)', 'External balance on goods and services (% of GDP)']
    
    df = df.loc[:, covariates]

    results = []
    for n in range(2, 10):
        km = KMeans(n_clusters=n).fit(df)
        results.append(km.inertia_)
    plt.plot(list(range(2, 10)), results)
    plt.xlabel("n_clusters")
    plt.ylabel("inertia")
    plt.title("Elbow-test for n_clusters")
    plt.show()

    km = KMeans(n_clusters=4).fit(df)
    preds = km.predict(df)
    plt.scatter(df.loc[:, 'Trade (% of GDP)'], df.loc[:, 'External balance on goods and services (% of GDP)'], c=preds)
    plt.xlabel("Trade (% of GDP)")
    plt.ylabel('External balance on goods and services (% of GDP)')
    plt.title("Clustering of countries by Trade (% of GDP) and External balance on goods and services (% of GDP)")
    # for i, txt in enumerate(countries):
    #     if txt in ["China", "Brazil", "Spain", "United States", "Australia", "Japan"]:
    #         plt.annotate(txt, (df.iloc[i, 0], df.iloc[i, 1]))
    plt.show()

# clustering()

def pca_clustering():
    df = prepare_df("wdi.csv")
    df = df[df["Year"] == 2015] # isolate to 2015
    countries = df.loc[:, "Country Name"]
    covariates = ['Trade (% of GDP)', 'External balance on goods and services (% of GDP)', 'Foreign direct investment, net inflows (% of GDP)', 'Employment in agriculture (% of total employment) (modeled ILO estimate)', 'Employment in industry (% of total employment) (modeled ILO estimate)','Employment in services (% of total employment) (modeled ILO estimate)']
    # covariates = ['Trade (% of GDP)', 'External balance on goods and services (% of GDP)']
    
    df = df.loc[:, covariates]
    df = pd.DataFrame(PCA(n_components=2).fit_transform(df))

    # results = []
    # for n in range(2, 10):
    #     km = KMeans(n_clusters=n).fit(df)
    #     results.append(km.inertia_)
    # plt.plot(list(range(2, 10)), results)
    # plt.xlabel("n_clusters")
    # plt.ylabel("inertia")
    # plt.title("Elbow-test for n_clusters")
    # plt.show()

    km = KMeans(n_clusters=5).fit(df)
    preds = km.predict(df)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=preds)
    plt.xlabel("Trade (% of GDP)")
    plt.ylabel('External balance on goods and services (% of GDP)')
    plt.title("Clustering of countries by Trade (% of GDP) and External balance on goods and services (% of GDP)")
    for i, txt in enumerate(countries):
        if txt in ["China", "Brazil", "Spain", "United States", "Australia", "Japan"]:
            plt.annotate(txt, (df.iloc[i, 0], df.iloc[i, 1]))
    plt.show()

# pca_clustering()

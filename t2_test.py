import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
from pysyncon import Dataprep, Synth
import cvxpy as cp


"""
Change file path (wdi.csv) to yours

runner lines are at the bottom

enable line 98 for the plots
"""

def diffInDiff(agreement):
    if agreement == "CPTPP":
        countries = ["Australia", "Canada", "Japan", "Mexico", "Vietnam"]
        implementation_year = 2020
    elif agreement == "CETA":
        countries = ["France", "Ireland", "Spain", "Italy", "Poland"]
        implementation_year = 2019
    else:
        countries = ["Gambia", "Ghana",  "Mali", "Nigeria", "Senegal"]
        implementation_year = 2016
    
    staging_df = pd.read_csv("wdi.csv")
    df = pd.DataFrame() 

    for yr in range(2008, 2024): # df is all of 2008 then all of 2009 etc.
        temp_df = pd.DataFrame()
        temp_df.insert(0, "Country Name", staging_df["Country Name"].unique())
        temp_df.insert(1, "Year", yr)
        for i, col in enumerate(staging_df["Series Name"].unique()):
            temp_df.insert(i+2, col, list(staging_df[staging_df["Series Name"] == col].loc[:, str(yr)]))
        df = pd.concat([df, temp_df])
        

    df.insert(len(df.columns), "Treatment", [1 if item in countries else 0 for item in df["Country Name"]])

    # scaler = StandardScaler() # scaling for some reason breaks it - can i scale for imputation, then unimpute
    # df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:]) 
    imp = KNNImputer(n_neighbors=5)  # <- take 5 most similar samples and impute absed on that 
    df.iloc[:, 1:] = imp.fit_transform(df.iloc[:, 1:])

    covariates = ['GDP (constant 2015 US$)', 'Trade (% of GDP)'
            ,'External balance on goods and services (% of GDP)']
    # covariates = ["GDP (constant 2015 US$)", "Trade (% of GDP)", "Exports of goods and services (constant 2015 US$)",
    # "Imports of goods and services (constant 2015 US$)", "Foreign direct investment, net inflows (% of GDP)", "Tariff rate, applied, simple mean, all products (%)",
    # "External balance on goods and services (% of GDP)", "Commercial service exports (current US$)", "Commercial service imports (current US$)", 
    # "Merchandise exports (current US$)", "Merchandise imports (current US$)", "GDP per capita (constant 2015 US$)", "Gini index"]


    logit = LogisticRegression()
    df['propensity_score'] = logit.fit(df[covariates], df['Treatment']).predict_proba(df[covariates])[:,1]

    # match treated units with control units
    treated = df[df['Treatment'] == 1]
    control = df[df['Treatment'] == 0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    matched_control_indices = indices.flatten()
    matched_control = control.iloc[matched_control_indices].reset_index(drop=True)
    matched_data = pd.concat([treated.reset_index(drop=True), matched_control], axis=0) # first 14 are countries, next 14 are their matches

    # for covariate in covariates:
    #     sns.kdeplot(matched_data[matched_data['Treatment'] == 1][covariate], color = (0.1, 0.3, 0.5, 0.2), label='Treated')
    #     sns.kdeplot(matched_data[matched_data['Treatment'] == 0][covariate], color = (0.5, 0.3, 0.1, 0.2), label='Control')
    #     plt.title(f'Balance check for {covariate}')
    #     plt.legend()
    #     plt.show()

    def did(df, df2, target):
        # Step 4: Apply Difference-in-Differences (DiD)
        df['Post'] = np.where(df['Year'] >= implementation_year, 1, 0)
        df['interaction'] = df['Treatment'] * df['Post']
        X = sm.add_constant(df[['Treatment', 'Post', 'interaction']])
        y = df[target]  # Outcome of interest (e.g., trade volume)
        DiD_model = sm.OLS(y, X).fit()
        # print(DiD_model.summary())

        if DiD_model.pvalues['interaction'] <= 0.05:
            t_ys = np.array(df2[df2["Treatment"] == 1].loc[:, ['Year', target]].groupby(['Year']).mean().iloc[:, 0])
            c_ys = np.array(df2[df2["Treatment"] == 0].loc[:, ['Year', target]].groupby(['Year']).mean().iloc[:, 0])
            plt.plot(list(range(2008, 2024)), t_ys, color='r', label='CPTPP Countries')
            plt.plot(list(range(2008, (2008 + len(c_ys)))), c_ys, color='b', label='non-CPTPP Countries')
            plt.vlines(2019, ymin=min(min(t_ys), min(c_ys)), ymax=max(max(t_ys), max(c_ys)), linestyle='--', color='gray', label='CPTPP Implementation')
            
            plt.title(f"{target} by year of CPTPP and non-CPTPP countries")
            plt.xlabel('Year')
            plt.ylabel(target)
            plt.legend()
            # plt.show()

        if DiD_model.pvalues['interaction'] <= 0.05:
            print(f"{target}={np.round(DiD_model.pvalues['interaction'], 3)}")
            return target
        else:
            return None

    targets = staging_df["Series Name"].unique()
    hits = []
    for x in targets:
        ans = did(matched_data, df, x)
        if ans != None:
            hits.append(ans)
    
    print(f"\n{len(hits)} variables are statistically significant\n")
    return(hits)

# targets = diffInDiff("CETA")
# targets = diffInDiff("CPTPP")



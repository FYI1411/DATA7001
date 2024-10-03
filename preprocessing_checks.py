import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from statsmodels.formula.api import ols, probit

"""
Import file, call prepare_df(file_path, True, True, True)

with your file path obviously

will return the pre-processed dataframe
"""

def prepare_df(file_path, anomaly=False, imputation=False, normalise=False):
    staging_df = pd.read_csv(file_path)
    df = pd.DataFrame() 
    for yr in range(2008, 2024): # df is all of 2008 then all of 2009 etc.
        temp_df = pd.DataFrame()
        temp_df.insert(0, "Country Name", staging_df["Country Name"].unique())
        temp_df.insert(1, "Year", yr)
        for i, col in enumerate(staging_df["Series Name"].unique()):
            temp_df.insert(i+2, col, list(staging_df[staging_df["Series Name"] == col].loc[:, str(yr)]))
        df = pd.concat([df, temp_df])
    
    if anomaly == True:
        original_nulls = df.isnull() # keep track of where the nulls were
        df.fillna(df.iloc[:, 2:].median(), inplace=True) # simple impute to find anomalies
        iso_forest = IsolationForest(contamination=0.3)  # 5% contamination
        y_pred = iso_forest.fit_predict(df.iloc[:, 2:])
        anomalies = (y_pred == -1)

        for col in df.iloc[:, 2:].columns:
            df.loc[anomalies, col] = np.nan # remove anomalies ready for imputation
        df[original_nulls] = np.nan # restore nulls

    if imputation == True:
        imp = KNNImputer(n_neighbors=5)  # <- take 5 most similar samples and impute absed on that 
        df.iloc[:, 1:] = imp.fit_transform(df.iloc[:, 1:])

    if normalise == True:
        scaler = StandardScaler()
        df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
    
    print(df)
    return df

def normalisation_test(df):
    # individual variable checks
    for col in df.iloc[:, 2:].columns:
        t_df = df[df["Year"] == 2015] # isolate to 2015
        t_df = t_df.dropna()
        values = list(t_df.loc[:, col])
        plt.hist(values)
        plt.title(f"Frequency histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    
    df = df.iloc[:, 2:]
    skewness = df.skew()
    kurtosis = df.kurtosis()
    mean_values = df.mean()

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Mean': mean_values
    })
    
    summary_mapping = {'GDP (constant 2015 US$)': 'GDP','Trade (% of GDP)': 'Trade %','Foreign direct investment, net inflows (% of GDP)': 'FDI In','Foreign direct investment, net outflows (% of GDP)': 'FDI Out','Tariff rate, applied, simple mean, all products (%)': 'Tariff %','Commercial service exports (current US$)': 'Comm. Exp.','Commercial service imports (current US$)': 'Comm. Imp.','Merchandise exports (current US$)': 'Merch. Exp.','Merchandise imports (current US$)': 'Merch. Imp.','GDP per capita (constant 2015 US$)': 'GDP/Cap.','Gini index': 'Gini','Agriculture, forestry, and fishing, value added (% of GDP)': 'Agri. %','Industry (including construction), value added (% of GDP)': 'Ind. %','Services, value added (% of GDP)': 'Serv. %','Employment in agriculture (% of total employment) (modeled ILO estimate)': 'Agri. Empl. %','Employment in industry (% of total employment) (modeled ILO estimate)': 'Ind. Empl. %','Employment in services (% of total employment) (modeled ILO estimate)': 'Serv. Empl. %','Exports of goods and services (constant LCU)': 'Exports','Imports of goods and services (constant 2015 US$)': 'Imports','External balance on goods and services (% of GDP)': 'Trade Balance %','Agricultural raw materials exports (% of merchandise exports)': 'Agri. Exp.','Agricultural raw materials imports (% of merchandise imports)': 'Agri. Imp.','Fuel exports (% of merchandise exports)': 'Fuel Exp.','Fuel imports (% of merchandise imports)': 'Fuel Imp.','Food exports (% of merchandise exports)': 'Food Exp.','Food imports (% of merchandise imports)': 'Food Imp.','Manufactures exports (% of merchandise exports)': 'Manu. Exp.','Manufactures imports (% of merchandise imports)': 'Manu. Imp.','Ores and metals exports (% of merchandise exports)': 'Ores Exp.','Ores and metals imports (% of merchandise imports)': 'Ores Imp.','Computer, communications and other services (% of commercial service exports)': 'Comp. Serv. Exp.','Computer, communications and other services (% of commercial service imports)': 'Comp. Serv. Imp.','Insurance and financial services (% of commercial service exports)': 'Ins. Serv. Exp.','Insurance and financial services (% of commercial service imports)': 'Ins. Serv. Imp.','Transport services (% of commercial service exports)': 'Trans. Serv. Exp.','Transport services (% of commercial service imports)': 'Trans. Serv. Imp.','Travel services (% of commercial service exports)': 'Trav. Serv. Exp.','Travel services (% of commercial service imports)': 'Trav. Serv. Imp.','Unemployment, total (% of total labor force) (national estimate)': 'Unempl. %','Poverty headcount ratio at national poverty lines (% of population)': 'Poverty %',}
    summary_df.index = summary_df.index.map(summary_mapping)
    summary_df = summary_df.sort_values(by='Mean', ascending=False)
    # Print the summary DataFrame
    # print(summary_df)
    # Step 4: Visualize the results
    plt.figure(figsize=(12, 8))
    # Plotting Skewness and Kurtosis

    sns.barplot(x=summary_df.index, y='Skewness', data=summary_df, palette='coolwarm')
    plt.title('Skewness of Each Feature')
    plt.xticks(rotation=90)
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

    sns.barplot(x=summary_df.index, y='Kurtosis', data=summary_df, palette='coolwarm')
    plt.title('Kurtosis of Each Feature')
    plt.xticks(rotation=90)
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

    # Plotting Mean values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=summary_df.index, y='Mean', data=summary_df, palette='viridis')
    plt.title('Mean Values of Each Feature')
    plt.xticks(rotation=90)
    plt.show()

def imputation_test(df):
    y = np.array(df.groupby("Country Name").apply(lambda x: x.isnull().sum().sum()))     # count NA's by country
    x = np.array(df.groupby('Country Name')['GDP (constant 2015 US$)'].mean())
    mask = ~np.isnan(x)  # Create a boolean mask for non-null x
    x = x[mask]    # Filter x using the mask
    y = y[mask]
    x = x.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
    pred = regr.predict(x) #
    mse = mean_squared_error(y, pred)  # Mean Squared Error
    r2 = r2_score(y, pred)             # R-squared (coefficient of determination)

    # Print the evaluation results
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    plt.scatter(x, y, color="black")
    plt.title("Number of missing values against GDP")
    plt.xlabel("GDP (constant 2015 US$)")
    plt.ylabel("Number of missing values")
    plt.plot(x, pred, color='red')

    plt.show()


# df = prepare_df("wdi.csv", anomaly=True, imputation=True, normalise=True)
# normalisation_test(df)
# imputation_test(df)



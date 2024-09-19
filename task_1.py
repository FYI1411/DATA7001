import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

temp_df = pd.read_csv("t1.csv")
df_imports = temp_df[temp_df['Series Name'] == 'Imports of goods and services (% of GDP)']
df_exports = temp_df[temp_df['Series Name'] == 'Exports of goods and services (% of GDP)']
df_trade = temp_df[temp_df['Series Name'] == 'Trade (% of GDP)']
df_merch = temp_df[temp_df['Series Name'] == 'Merchandise exports (current US$)']
df_serv = temp_df[temp_df['Series Name'] == 'Commercial service exports (current US$)']

df = pd.DataFrame()
df.insert(0, "Country Code", temp_df.iloc[:, 0].unique())
df.insert(1, "Imports (% of GDP)", np.array(df_imports.iloc[:, -1]))
df.insert(2, "Exports (% of GDP)", np.array(df_exports.iloc[:, -1]))
df.insert(3, "Trade (% of GDP)", np.array(df_trade.iloc[:, -1]))
df.insert(4, "Merchandise Exports (current US$)", np.array(df_merch.iloc[:, -1]))
df.insert(5, "Commercial service exports (current US$)", np.array(df_serv.iloc[:, -1]))

# print(df.isnull().sum()) # in this case, we can just drop any nulls 
df = df.dropna()
# print(df.info()) # leaves 165 points

""" transform data into preferred format (code, trade balance (% of GDP), trade (% of GDP), 
    merchandise/commercial service), then scaled """
df1 = pd.DataFrame()
# df1.insert(0, "Country Code", df.iloc[:, 0])
df1.insert(0, "Trade balance (% of GDP)", np.array(df.iloc[:, 1])/np.array(df.iloc[:, 2]))
df1.insert(1, "Trade (% of GDP)", np.array(df.iloc[:, 3]))
df1.insert(2, "Merchandise/Commercial Exports", np.array(df.iloc[:, 4])/np.array(df.iloc[:, 5]))

scaler = StandardScaler()
df1 = pd.DataFrame(scaler.fit_transform(df1)) # scale + normalise
df1 = pd.DataFrame(normalize(df1))


km = KMeans(n_clusters=3).fit(df1.iloc[:, :2])
preds = km.predict(df1.iloc[:, :2])
plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], c=preds)
plt.xlabel('Trade balance (% of GDP)')
plt.ylabel('Trade (% of GDP)')
# for i, txt in enumerate(np.array(df.iloc[:, 0])):
#     plt.annotate(txt, (df1.iloc[i, 0], df1.iloc[i, 1]))

plt.show()

km = KMeans(n_clusters=3).fit(df1.iloc[:, :3])
preds = km.predict(df1.iloc[:, :3])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df1.iloc[:, 0], df1.iloc[:, 1], df1.iloc[:, 2], c=preds)
ax.set_xlabel("Trade balance (% of GDP)")
ax.set_ylabel("Trade (% of GDP)")
ax.set_zlabel('Merchandise/Commercial Exports')


plt.show()
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.api import Series
import numpy as np
import scipy.stats as stats
from matplotlib.lines import Line2D

df = pd.read_csv('wdi.csv')
print(df.columns)
series_names = [
 #'GDP (constant 2015 US$)' 'Trade (% of GDP)'
 #'Foreign direct investment, net inflows (% of GDP)'
 #'Foreign direct investment, net outflows (% of GDP)'
 #'Tariff rate, applied, simple mean, all products (%)'
 #'Commercial service exports (current US$)',
 #'Commercial service imports (current US$)',
 #'Merchandise exports (current US$)',
 #'Merchandise imports (current US$)'
 #'GDP per capita (constant 2015 US$)' 'Gini index'
 #'Agriculture, forestry, and fishing, value added (% of GDP)'
 #'Industry (including construction), value added (% of GDP)'
 #'Services, value added (% of GDP)'
 #'Employment in agriculture (% of total employment) (modeled ILO estimate)'
 #'Employment in industry (% of total employment) (modeled ILO estimate)'
 #'Employment in services (% of total employment) (modeled ILO estimate)'
 #'Exports of goods and services (constant LCU)'
 #'Imports of goods and services (constant 2015 US$)'
 #'External balance on goods and services (% of GDP)'
 #'Agricultural raw materials exports (% of merchandise exports)',
 'Agricultural raw materials imports (% of merchandise imports)',
 #'Fuel exports (% of merchandise exports)',
 'Fuel imports (% of merchandise imports)',
 #'Food exports (% of merchandise exports)',
 'Food imports (% of merchandise imports)',
 #'Manufactures exports (% of merchandise exports)',
 'Manufactures imports (% of merchandise imports)',
 #'Ores and metals exports (% of merchandise exports)',
 'Ores and metals imports (% of merchandise imports)'
 #'Computer, communications and other services (% of commercial service exports)',
 #'Computer, communications and other services (% of commercial service imports)',
 #'Insurance and financial services (% of commercial service exports)',
 #'Insurance and financial services (% of commercial service imports)',
 #'Transport services (% of commercial service exports)',
 #'Transport services (% of commercial service imports)',
 #'Travel services (% of commercial service exports)',
 #'Travel services (% of commercial service imports)',
 #'Unemployment, total (% of total labor force) (national estimate)'
 #'Poverty headcount ratio at national poverty lines (% of population)'
]
#series_names = [
#     'Merchandise exports (current US$)',
#     'Commercial service exports (current US$)'
#      'Agricultural raw materials exports (% of merchandise exports)',
#      'Fuel exports (% of merchandise exports)',
#      'Food exports (% of merchandise exports)',
#      'Manufactures exports (% of merchandise exports)',
#      'Ores and metals exports (% of merchandise exports)',
#    'Computer, communications and other services (% of commercial service exports)',
#     'Insurance and financial services (% of commercial service exports)',
#     'Transport services (% of commercial service exports)',
#     'Travel services (% of commercial service exports)'
#]
'''
plt.style.use('seaborn-v0_8')

# Assuming df is already defined and filtered
df = df[df['Country Name'] == 'Australia']
print("Unique Series Names:", df['Series Name'].unique())

# Filter the DataFrame for these specific series names
filtered_data = df[df['Series Name'].isin(series_names)]

# Set the 'Series Name' as index and drop unnecessary columns
filtered_data.set_index('Series Name', inplace=True)
filtered_data = filtered_data.drop(columns=['Country Name'])

# Transpose the DataFrame to have years as index
filtered_data = filtered_data.T

# Convert the filtered data to numeric if necessary
filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')

#Calculate averages for 2014-2018 and 2019-2023
averages_df = filtered_data.loc['2013':]
'''
plt.style.use('seaborn-v0_8')

# Filter the DataFrame for Australia
df = df[df['Country Name'] == 'Australia']
print("Unique Series Names:", df['Series Name'].unique())

# Filter the DataFrame for the specific series names
filtered_data = df[df['Series Name'].isin(series_names)]

# Set 'Series Name' as index and drop unnecessary columns
filtered_data.set_index('Series Name', inplace=True)
filtered_data = filtered_data.drop(columns=['Country Name'])

# Transpose the DataFrame to have years as index
filtered_data = filtered_data.T

# Convert the filtered data to numeric if necessary
filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')

rename_mapping = {
    'Computer, communications and other services (% of commercial service exports)': 'Computer, communications and other services',
    'Insurance and financial services (% of commercial service exports)': 'Insurance and financial services',
    'Transport services (% of commercial service exports)': 'Transport services',
    'Travel services (% of commercial service exports)': 'Travel services'
}

rename_mapping = {
    'Agricultural raw materials imports (% of merchandise imports)': 'Agricultural raw materials imports',
    'Fuel imports (% of merchandise imports)': 'Fuel imports',
    'Food imports (% of merchandise imports)': 'Food imports',
    'Manufactures imports (% of merchandise imports)': 'Manufactures imports',
    'Ores and metals imports (% of merchandise imports)': 'Ores and metals imports'
}

# Rename the columns using the mapping
filtered_data.rename(columns=rename_mapping, inplace=True)

# Select only the years from 2014 onwards
averages_df = filtered_data.loc['2014':]

# Calculate averages for the periods 2014-2018 and 2019-2023
averaged_periods = {
    'Before CPTPP': averages_df.loc['2014':'2018'].mean(),
    'After CPTPP': averages_df.loc['2019':'2023'].mean()
}

# Create a DataFrame from the averaged periods
averaged_df = pd.DataFrame(averaged_periods).T

# Sort the columns by the average value of 2014-2018
averaged_df = averaged_df.sort_values(by='Before CPTPP', axis=1, ascending=False)

ax = averaged_df.plot(kind='bar')

# Customize the plot
plt.title('Average Australian Merchandise Imports (2014-2018 vs 2019-2023)')
plt.ylabel('Percentage of merchandise imports (%)')
#plt.xlabel('Year')
plt.xticks(rotation=0)
plt.tight_layout()
#ax.axvline(x=5.5, color='red', linestyle='--', label='CPTPP Signed')  # Adjust x=5 if needed based on your data's structure
plt.legend(fontsize=10, loc='best')
# Add labels to each bar
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:  # Add label only for non-zero bars
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
                height + 0.5,  # Y position (just above the bar)
                f'{height:.2f}',  # Label format
                ha='center', va='bottom', fontsize=9, color='black'  # Adjust label appearance
            )
plt.show()
'''
# Split the list into b_scores and i_scores
split_index = len(values) // 2  # Or set it to 5 if you prefer
b_scores = values[:split_index]  # Pre scores
i_scores = values[split_index:]   # Post scores

# Example of how you might integrate this into your plotting function
def individual_combined_bars(b_scores, i_scores) -> None:
        b_e, i_e = stats.sem(b_scores, ddof=1), stats.sem(i_scores, ddof=1) # gets standard error of both sets (which is like the highlighted section)
        c = (i_scores.mean()-i_e) - (b_scores.mean()+b_e) # stat sig test for 2-sample t-test
        # basically, if the highlighted regions touch, c < 1 and the result is NOT stat sig
        # this code will then highlight in RED (otherwise green)
        colour = "red" if c < 0 else "green"
        fig, axis = plt.subplots(figsize=(6, 4)) 
        axis.set_title('Australia Ore, Metal And Fuel Exports Pre/Post CPTPP')
        # bar is the highlighted section, which goes from mean - SE to mean + SE for both
        axis.bar(len(b_scores)/2, 2*b_e, len(b_scores), bottom=b_scores.mean()-b_e, color=[0,0,1,0.3])
        # hline at the mean
        axis.hlines(y=b_scores.mean(),color=[0,0,1,0.7], xmin=0, xmax=len(b_scores))
        axis.bar(7, 2*i_e, 4, bottom=i_scores.mean()-i_e, color=[0,1,0,0.3])
        axis.hlines(y=i_scores.mean(), xmin=5, xmax=9, color=[0,1,0,0.7])
        
        # plot each point, and then scatterplot ontop to make it looks nice
        axis.plot(np.arange(len(b_scores)+1), np.concatenate((b_scores, i_scores[0]), axis=None), color="blue")
        axis.plot(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.scatter(np.arange(len(b_scores)), b_scores, color="blue")
        axis.scatter(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.set(xticks = np.arange(0, len(b_scores)+len(i_scores),3)) 
        #axis.set_xlim(-0.5, 1.5)

        # for the legend, just ignore
        axis.legend([Line2D([0], [0], color=[0,0,1,1], lw=4), Line2D([0], [0], color=[0,1,0,1], lw=4)], ["Pre CPTPP", "Post CPTPP"])

        axis.set_xticks([i for i in range(10)])
        axis.set_xticklabels([str(yr) for yr in range(2014, 2024)])
        plt.ylabel('Ores, Metal & Fuel % Of Merchandise Exports (Z-score)')
        plt.xlabel('year')
        plt.show()

individual_combined_bars(b_scores, i_scores)
'''

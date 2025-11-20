#### FUZZY MATCHING ON NAMES, EXACT JOINING ON GENDER AND AGE GROUP####
#Libraries
import pandas as pd
import jellyfish
from thefuzz import fuzz
import itertools
import numpy as np
import plotly.express as px
import statsmodels

# File paths
filepath_marathon = r'Data\2024_ChicagoMarathonResults.csv'
filepath_parkrun = r'Data\Cornerstonelakes_event77.csv'

#import data
marathon_df = pd.read_csv(filepath_marathon)
parkrun_df = pd.read_csv(filepath_parkrun)

print('Number of people in marathon DF: ',len(marathon_df))
print('Number of unique names in marathon DF: ',len(marathon_df['Name'].unique()))
print('Number of people in parkrun DF: ',len(parkrun_df))
print('Number of unique names in parkrun DF: ',len(parkrun_df['Name'].unique()))

# Normalise name
parkrun_df['Name'] = parkrun_df['Name'].str.lower().str.strip() #Convert to lowercase & remove leading/trailing whitespace
parkrun_df = parkrun_df[parkrun_df['Name'] != 'unknown'] #Remove unknowns
marathon_df['Name'] = marathon_df['Name'].str.lower().str.strip() #Convert to lowercase & remove leading/trailing whitespace


# Normalise gender
parkrun_df['Gender'] = parkrun_df['Gender'].str.lower().str.strip() #Convert to lowercase & remove leading/trailing whitespace
marathon_df['Gender'] = marathon_df['Gender'].replace({
    'Men': 'male',
    'Women': 'female'
})

# Normalise age group
parkrun_df['Age Group'] = parkrun_df['Age group'].str.extract(r'(\d{2}-\d{2})') #Format data
parkrun_df['Age Group'] = parkrun_df['Age Group'].replace({
    '80-84':'80+',
    '11-14':'Under 20'
})


###FUZZY MATCHING
# create index
parkrun_df = parkrun_df.reset_index(drop=False)
parkrun_df.rename(columns={'index': 'Parkrun_ID'}, inplace=True)
marathon_df = marathon_df.reset_index(drop=False)
marathon_df.rename(columns={'index': 'Marathon_ID'}, inplace=True)

all_pairs = list(itertools.product(parkrun_df['Parkrun_ID'], marathon_df['Marathon_ID']))
df_pairs = pd.DataFrame(all_pairs, columns=['Parkrun_ID', 'Marathon_ID'])
print("Number of pairs:", len(df_pairs)) # 7907344

# Merge once with parkrun_df and marathon_df
df_merge = (
    df_pairs
    .merge(parkrun_df, on='Parkrun_ID', how='left')
    .merge(marathon_df, on='Marathon_ID', how='left')
)
print("Number of pairs:", len(df_merge)) # 7907344

# Rename Name_x and Name_y after merge
df_merge.rename(columns={'Name_x': 'Parkrun Name',
                         'Name_y': 'Marathon Name',
                         'Age Group_y': 'Age Group',
                         'Age Group_x': 'Parkrun Age',
                         'Gender_x':'Parkrun Gender',
                         'Gender_y':'Gender'}, inplace=True)


# Levenshtein Score (using TheFuzz python library)
df_merge['Levenshtein Score'] = df_merge.apply(
    lambda row: fuzz.partial_ratio(row['Parkrun Name'], row['Marathon Name']),
    axis=1
)

# Compute Jaro-Winkler similarity (using jellyfish library)
df_merge['Jaro-Winkler Score'] = df_merge.apply(
    lambda row: jellyfish.jaro_winkler_similarity(row['Parkrun Name'], row['Marathon Name']),
    axis=1
)


# Filter where Age Group and Gender match
filtered = df_merge[
    (df_merge['Parkrun Age'] == df_merge['Age Group']) &
    (df_merge['Parkrun Gender'] == df_merge['Gender'])
]
print(len(filtered['Parkrun Name'].unique()))

# Sort by Parkrun Name and score (prioritise higher scores)
filtered_sorted = filtered.sort_values(by=['Parkrun Name', 'Levenshtein Score'], ascending=[True, False])
levenshtein_results = filtered_sorted[['Parkrun Name', 'Marathon Name', 'Levenshtein Score', 'Parkrun Age', 'Age Group', 'Parkrun Gender', 'Gender','Time','Finish']]
jaro_winkler_results = filtered_sorted.sort_values(by=['Parkrun Name', 'Jaro-Winkler Score'], ascending=[True, False])
jaro_winkler_results = jaro_winkler_results[['Parkrun Name', 'Marathon Name', 'Jaro-Winkler Score', 'Parkrun Age', 'Age Group', 'Parkrun Gender', 'Gender','Time','Finish']]

# Restrict results set
# Levenshtein
filtered_levenshtein = (
    levenshtein_results.groupby('Parkrun Name', group_keys=False)
    .apply(lambda g: g[g['Levenshtein Score'] == 100] if (g['Levenshtein Score'] == 100).any() else g.nlargest(5, 'Levenshtein Score'))
    .reset_index(drop=True)
)

# Jaro_winkler
filtered_jaro_winkler = (
    jaro_winkler_results.groupby('Parkrun Name', group_keys=False)
    .apply(lambda g: g[g['Jaro-Winkler Score'] == 1] if (g['Jaro-Winkler Score'] == 1).any() else g.nlargest(5, 'Jaro-Winkler Score'))
    .reset_index(drop=True)
)
# Deduplicating data
#Restrict dataset to just top similarity score
final_levenshtein = filtered_levenshtein = (
    levenshtein_results.groupby('Parkrun Name', group_keys=False)
    .apply(lambda g: g[g['Levenshtein Score'] == 100] if (g['Levenshtein Score'] == 100).any() else g.nlargest(1, 'Levenshtein Score'))
    .reset_index(drop=True)
)

final_jaro_winkler = (
    jaro_winkler_results.groupby('Parkrun Name', group_keys=False)
    .apply(lambda g: g[g['Jaro-Winkler Score'] == 1] if (g['Jaro-Winkler Score'] == 1).any() else g.nlargest(1, 'Jaro-Winkler Score'))
    .reset_index(drop=True)
)


# Identify duplicated rows based on the specified columns
# Define the columns to check for duplicates
duplicate_cols_marathon = [
    'Marathon Name', 'Age Group', 'Gender'
]
duplicate_cols_parkrun = [
    'Parkrun Name', 'Parkrun Age', 'Parkrun Gender'
]

# Count duplicates (excluding the first occurrence)
num_duplicates_L = final_levenshtein.duplicated(subset=duplicate_cols_marathon, keep='first').sum()
num_duplicates_JW = final_jaro_winkler.duplicated(subset=duplicate_cols_marathon, keep='first').sum()

print("Number of duplicate rows (Levenschtein) for marathon data:", num_duplicates_L)
print("Number of duplicate rows (Jaro-Winkler) for marathon data:", num_duplicates_JW)

# Count duplicates (excluding the first occurrence)
num_duplicates_L = final_levenshtein.duplicated(subset=duplicate_cols_parkrun, keep='first').sum()
num_duplicates_JW = final_jaro_winkler.duplicated(subset=duplicate_cols_parkrun, keep='first').sum()

print("Number of duplicate rows (Levenschtein) for parkrun data:", num_duplicates_L)
print("Number of duplicate rows (Jaro-Winkler) for parkrun data:", num_duplicates_JW)

#Create dataframe with duplicates
duplicates_df = final_levenshtein[final_levenshtein.duplicated(subset=duplicate_cols_marathon, keep=False)]

# Remove duplicates
# Sort by Levenshtein Score so the highest comes first
final_levenshtein_sorted = final_levenshtein.sort_values(by='Levenshtein Score', ascending=False)

# Mark duplicates (keep first occurrence which is highest score after sorting)
mask_lowest = final_levenshtein_sorted.duplicated(subset=duplicate_cols_marathon, keep='first')

cols_to_zero = ['Marathon Name', 'Levenshtein Score', 'Age Group', 'Gender', 'Finish']

# Apply zeroing
final_levenshtein_sorted.loc[mask_lowest, cols_to_zero] = 0



# Validation
# Number of exact matches in the data
# Count the number of rows where 'Levenshtein Score' equals 100
count_levenshtein_100 = (final_levenshtein_sorted['Levenshtein Score'] == 100).sum()
count_JaroWinkler_100 = (final_jaro_winkler['Jaro-Winkler Score'] == 1).sum()
print("Number of records with Levenshtein Score = 100:", count_levenshtein_100)
print("Number of records with Jaro-Winkler Score = 1:", count_JaroWinkler_100)

# Decision
# Define conditions
conditions_JW = [
    final_jaro_winkler['Jaro-Winkler Score'] > 0.9,
    (final_jaro_winkler['Jaro-Winkler Score'] <= 0.9) & (final_jaro_winkler['Jaro-Winkler Score'] > 0.8)
]
conditions_L = [
    final_levenshtein_sorted['Levenshtein Score'] > 90,
    (final_levenshtein_sorted['Levenshtein Score'] <= 90) & (final_levenshtein_sorted['Levenshtein Score'] > 80)
]

# Define corresponding choices
choices = ['Accept', 'Manual review']

# Apply conditions
final_jaro_winkler['Decision'] = np.select(conditions_JW, choices, default='Reject')
final_levenshtein_sorted['Decision'] = np.select(conditions_L, choices, default='Reject')


# Create visualisation
# Jaro Winkler
# Count occurrences of each decision
decision_counts = final_jaro_winkler['Decision'].value_counts().reset_index()
decision_counts.columns = ['Decision', 'Count']


# Define a consistent color mapping for decisions
color_map = {
    'Accept': 'green',
    'Manual review': 'orange',
    'Reject': 'red'
}

# Create pie chart
fig = px.pie(
    decision_counts,
    names='Decision',
    values='Count',
    title='Distribution of Decisions using Jaro-Winkler scores',
    hole=0.3,  # makes it a donut chart
    color='Decision',  # Use Decision for color
    color_discrete_map=color_map  # Apply consistent colors
)

# Show the chart
fig.show()

# Levenshtein Scores
# Count occurrences of each decision
decision_counts = final_levenshtein_sorted['Decision'].value_counts().reset_index()
decision_counts.columns = ['Decision', 'Count']

# Create pie chart
fig = px.pie(
    decision_counts,
    names='Decision',
    values='Count',
    title='Distribution of Decisions using Levenshtein scores',
    hole=0.3,  # makes it a donut chart
    color='Decision',  # Use Decision for color
    color_discrete_map=color_map  # Apply consistent colors
)

# Show the chart
fig.show()

#Insights

#final_levenshtein_sorted
#final_jaro_winkler

filtered_df = final_jaro_winkler[final_jaro_winkler['Decision'] == 'Accept'].copy()

# Clean 'Time' column (Parkrun completion time)
# Remove trailing ':00' if present (e.g., '25:48:00' -> '25:48')
filtered_df['Time'] = filtered_df['Time'].str.replace(r':00$', '', regex=True)

# If value does not contain ':', append ':00'
filtered_df['Time'] = filtered_df['Time'].apply(lambda x: x + ':00' if ':' not in x else x)

# If value does not contain ':', append '00:' to start
filtered_df['Time'] = filtered_df['Time'].apply(lambda x: '00:' + x if x.count(':') == 1 else x)

# Convert to time
filtered_df['Time'] = pd.to_timedelta(filtered_df['Time'])

# Clean 'Finish' column (Marathon completion time)
filtered_df['Finish'] = pd.to_timedelta(filtered_df['Finish'])

# Convert timedelta to total minutes
filtered_df['Finish_minutes'] = filtered_df['Finish'].dt.total_seconds() / 3600
filtered_df['Time_minutes'] = filtered_df['Time'].dt.total_seconds() / 60

# Create scatter plot using numeric values (minutes)
fig = px.scatter(
    filtered_df,
    x='Finish_minutes',
    y='Time_minutes',
    title='Parkrun vs Marathon Completion Times (Accepted Matches)',
    labels={'Finish_minutes': 'Marathon completion time (hours)', 'Time_minutes': 'Parkrun completion time (minutes)'},
    hover_data=['Marathon Name', 'Parkrun Name', 'Finish', 'Time'],
    trendline='ols',  # Adds regression line
    trendline_color_override='red'  # Optional: make the line stand out

)

# Format axis ticks to show minutes clearly
fig.update_xaxes(title='Marathon completion time (hours)')
fig.update_yaxes(title='Parkrun completion time (minutes)')

fig.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1 objective - To load and preprocess the PMGSY rural infrastructure dataset, ensuring data types are appropriate
#and missing values are handled for smooth analysis.


df = pd.read_csv(r"C:\Users\Alekhya\Downloads\d4361151-6d41-43c7-98cd-9a6cd90b5ca4 (8).csv")


print("First 5 Rows:\n", df.head())




print("Before Preprocessing:")
print(df.info())
print("\nMissing values before handling:\n", df.isnull().sum())
df = df.drop_duplicates()

# Convert numerical columns to correct data types
num_cols = [
    'NO_OF_ROAD_WORK_SANCTIONED', 'NO_OF_BRIDGES_SANCTIONED',
    'NO_OF_ROAD_WORKS_COMPLETED', 'NO_OF_BRIDGES_COMPLETED',
    'NO_OF_ROAD_WORKS_BALANCE', 'NO_OF_BRIDGES_BALANCE',
    'LENGTH_OF_ROAD_WORK_SANCTIONED_KM', 'COST_OF_WORKS_SANCTIONED_LAKHS',
    'LENGTH_OF_ROAD_WORK_COMPLETED_KM', 'EXPENDITURE_OCCURED_LAKHS',
    'LENGTH_OF_ROAD_WORK_BALANCE_KM'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts any wrong types

# Handle missing numeric values using median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Handle missing values in categorical columns
df['STATE_NAME'] = df['STATE_NAME'].fillna(df['STATE_NAME'].mode()[0])
df['DISTRICT_NAME'] = df['DISTRICT_NAME'].fillna(df['DISTRICT_NAME'].mode()[0])
df['PMGSY_SCHEME'] = df['PMGSY_SCHEME'].fillna(df['PMGSY_SCHEME'].mode()[0])

# Show summary after preprocessing
print("\nAfter Preprocessing:")
print(df.info())
print("\nMissing values after handling:\n", df.isnull().sum())

# Save cleaned dataset (optional)
df.to_csv("pmgsy_cleaned.csv", index=False)

#2nd objective To perform exploratory data analysis (EDA) to understand the distribution and trends of key variables such as
#Sanctioned Length, Completed Length, and Balance Length of roads and bridges.

#creates a new column in your DataFrame called COMPLETION_RATIO.

cols_to_convert = [
    'LENGTH_OF_ROAD_WORK_SANCTIONED_KM',
    'LENGTH_OF_ROAD_WORK_COMPLETED_KM',
    'LENGTH_OF_ROAD_WORK_BALANCE_KM',
    'COST_OF_WORKS_SANCTIONED_LAKHS',
    'EXPENDITURE_OCCURED_LAKHS'
]


df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in important columns
df.dropna(subset=cols_to_convert, inplace=True)

# Create Completion Ratio
df['COMPLETION_RATIO'] = df['LENGTH_OF_ROAD_WORK_COMPLETED_KM'] / df['LENGTH_OF_ROAD_WORK_SANCTIONED_KM']

# Set Seaborn theme
sns.set(style="whitegrid")

# -------------------------------
# 1. Distribution Plots for Road Lengths
# -------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['LENGTH_OF_ROAD_WORK_SANCTIONED_KM'], kde=True, color='skyblue')
plt.title('Distribution of Sanctioned Road Length (KM)')

plt.subplot(1, 3, 2)
sns.histplot(df['LENGTH_OF_ROAD_WORK_COMPLETED_KM'], kde=True, color='green')
plt.title('Distribution of Completed Road Length (KM)')

plt.subplot(1, 3, 3)
sns.histplot(df['LENGTH_OF_ROAD_WORK_BALANCE_KM'], kde=True, color='orange')
plt.title('Distribution of Balance Road Length (KM)')

plt.tight_layout()
plt.show()


# 2. Bar plot - Top 10 states by completed road length
plt.figure(figsize=(12, 6))
top_states = df.groupby('STATE_NAME')['LENGTH_OF_ROAD_WORK_COMPLETED_KM'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
plt.title("Top 10 States by Completed Road Length")
plt.xlabel("Completed Length (in km)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# 3. Scatter plot - Sanctioned vs Completed Road Length
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='LENGTH_OF_ROAD_WORK_SANCTIONED_KM',
    y='LENGTH_OF_ROAD_WORK_COMPLETED_KM',
    hue='STATE_NAME',
    alpha=0.7,
    legend=False
)
plt.title("Sanctioned vs Completed Road Length")
plt.xlabel("Sanctioned Length (km)")
plt.ylabel("Completed Length (km)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Heatmap - Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[cols_to_convert + ['COMPLETION_RATIO']].corr(), annot=True, cmap='YlGnBu')
plt.title("Correlation Matrix of Key Metrics")
plt.tight_layout()
plt.show()

# 5. Pie Chart - Completed vs Balance Road Work
plt.figure(figsize=(6, 6))
completed = df['LENGTH_OF_ROAD_WORK_COMPLETED_KM'].sum()
balance = df['LENGTH_OF_ROAD_WORK_BALANCE_KM'].sum()
plt.pie(
    [completed, balance],
    labels=['Completed', 'Balance'],
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999'],
    startangle=140
)
plt.title("Overall Road Work Status")
plt.tight_layout()
plt.show()

# 6. Box Plot - Outlier Detection in Sanctioned Cost
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['COST_OF_WORKS_SANCTIONED_LAKHS'], color='orange')
plt.title("Outliers in Sanctioned Cost (Lakhs)")
plt.xlabel("Cost of Work Sanctioned (in Lakhs)")
plt.tight_layout()




# -------------------------------
# 7. Box Plots to Detect Outliers
# -------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['LENGTH_OF_ROAD_WORK_SANCTIONED_KM', 'LENGTH_OF_ROAD_WORK_COMPLETED_KM', 'LENGTH_OF_ROAD_WORK_BALANCE_KM']])
plt.title('Boxplot of Road Work Lengths (KM)')
plt.ylabel("Length in KM")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 8. Trend Across States (Top 10 States)
# -------------------------------
top_states = df.groupby('STATE_NAME')[['LENGTH_OF_ROAD_WORK_SANCTIONED_KM', 'LENGTH_OF_ROAD_WORK_COMPLETED_KM', 'LENGTH_OF_ROAD_WORK_BALANCE_KM']].sum().sort_values(by='LENGTH_OF_ROAD_WORK_SANCTIONED_KM', ascending=False).head(10)

top_states.plot(kind='bar', figsize=(15, 6), colormap='tab10')
plt.title('Top 10 States by Sanctioned, Completed, and Balance Road Length')
plt.ylabel('Length (KM)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#3RD Objective To analyze the relationship between sanctioned works and completion rates, using scatter plots and
#correlation heatmaps to identify efficiency gaps across regions
# Calculate Completion Rates
df['ROAD_COMPLETION_RATE'] = df['NO_OF_ROAD_WORKS_COMPLETED'] / df['NO_OF_ROAD_WORK_SANCTIONED']
df['BRIDGE_COMPLETION_RATE'] = df['NO_OF_BRIDGES_COMPLETED'] / df['NO_OF_BRIDGES_SANCTIONED']

# Replace infinities and NaNs (in case of division by zero)
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.fillna(0, inplace=True)

# -------------------------------
# 1. Scatter Plots: Sanctioned vs Completed
# -------------------------------
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='NO_OF_ROAD_WORK_SANCTIONED', y='ROAD_COMPLETION_RATE', hue='STATE_NAME', data=df, legend=False)
plt.title('Road Completion Rate vs Sanctioned Road Works')
plt.xlabel('Road Works Sanctioned')
plt.ylabel('Road Completion Rate')

plt.subplot(1, 2, 2)
sns.scatterplot(x='NO_OF_BRIDGES_SANCTIONED', y='BRIDGE_COMPLETION_RATE', hue='STATE_NAME', data=df, legend=False)
plt.title('Bridge Completion Rate vs Sanctioned Bridge Works')
plt.xlabel('Bridge Works Sanctioned')
plt.ylabel('Bridge Completion Rate')

plt.tight_layout()
plt.show()

# -------------------------------
# 2. Correlation Heatmap
# -------------------------------
correlation_columns = [
    'NO_OF_ROAD_WORK_SANCTIONED', 'NO_OF_ROAD_WORKS_COMPLETED', 'ROAD_COMPLETION_RATE',
    'NO_OF_BRIDGES_SANCTIONED', 'NO_OF_BRIDGES_COMPLETED', 'BRIDGE_COMPLETION_RATE'
]

corr_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Correlation Heatmap - Sanctioned vs Completion Rates')
plt.show()
#4th objective-To visualize the performance of different states using bar plots,
#highlighting areas with high completion or backlog of projects.
# Group by state and calculate total sanctioned, completed, and balance road works
state_summary = df.groupby('STATE_NAME').agg({
    'NO_OF_ROAD_WORK_SANCTIONED': 'sum',
    'NO_OF_ROAD_WORKS_COMPLETED': 'sum',
    'NO_OF_ROAD_WORKS_BALANCE': 'sum'
}).reset_index()

# Sort for better visualization
state_summary = state_summary.sort_values(by='NO_OF_ROAD_WORK_SANCTIONED', ascending=False)

# Set style
plt.figure(figsize=(14, 7))
bar_width = 0.25
x = range(len(state_summary))

# Plotting
plt.bar(x, state_summary['NO_OF_ROAD_WORK_SANCTIONED'], width=bar_width, label='Sanctioned', color='skyblue')
plt.bar([p + bar_width for p in x], state_summary['NO_OF_ROAD_WORKS_COMPLETED'], width=bar_width, label='Completed', color='seagreen')
plt.bar([p + bar_width*2 for p in x], state_summary['NO_OF_ROAD_WORKS_BALANCE'], width=bar_width, label='Balance', color='salmon')

# Axis and labels
plt.xticks([p + bar_width for p in x], state_summary['STATE_NAME'], rotation=90)
plt.ylabel("Number of Road Works")
plt.title("State-wise Performance: Sanctioned vs Completed vs Balance")
plt.legend()
plt.tight_layout()
plt.show()

# 5th objective -To detect outliers and inconsistencies in sanctioned vs completed work
#using box plots and distribution plots to ensure data quality and reliability.
# Set style
sns.set(style="whitegrid")

# Box plots to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['LENGTH_OF_ROAD_WORK_SANCTIONED_KM', 'LENGTH_OF_ROAD_WORK_COMPLETED_KM']])
plt.title("Box Plot - Sanctioned vs Completed Road Work Length")
plt.ylabel("Length (KM)")
plt.xticks([0, 1], ['Sanctioned Length', 'Completed Length'])
plt.show()

# Distribution plots to check spread and potential inconsistencies
plt.figure(figsize=(12, 6))
sns.histplot(df['LENGTH_OF_ROAD_WORK_SANCTIONED_KM'], kde=True, color='blue', label='Sanctioned', bins=30)
sns.histplot(df['LENGTH_OF_ROAD_WORK_COMPLETED_KM'], kde=True, color='green', label='Completed', bins=30)
plt.title("Distribution - Sanctioned vs Completed Road Work Length")
plt.xlabel("Length (KM)")
plt.legend()
plt.show()


#6th objective- To categorize and visualize the completion percentage of sanctioned projects using pie charts to
#show proportional success rates across regions.

# Selecting relevant columns for correlation analysis
corr_columns = [
    'LENGTH_OF_ROAD_WORK_SANCTIONED_KM',
    'LENGTH_OF_ROAD_WORK_COMPLETED_KM',
    'LENGTH_OF_ROAD_WORK_BALANCE_KM',
    'COST_OF_WORKS_SANCTIONED_LAKHS',
    'EXPENDITURE_OCCURED_LAKHS'
]

# Compute correlation matrix
correlation_matrix = df[corr_columns].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("Correlation Heatmap of Road Work Metrics")
plt.show()

# Optional: Show covariance matrix as well
covariance_matrix = df[corr_columns].cov()
print("Covariance Matrix:\n", covariance_matrix)

#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction

# ## Import Libraries And Load Dataset
# 

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[29]:


df = pd.read_csv('data.csv')


# ## Data Wrangling/Preprocessing
# 
# ###  Data Exploration and Cleaning
# Explored the dataset to understand its structure, feature distribution, and potential insights.
# 
# Handled missing values, outliers, or any other data inconsistencies to ensure a clean dataset for analysis.

# In[5]:


df.shape


# In[6]:


## Display information about the dataset
df.info()


# In[30]:


## Display rows of the dataset
df.head()


#  ### Data Cleaning
#  
#  Removing the unnecessary column with missing values

# In[31]:


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# In[32]:


# Removing the unnecessary column . Unnamed: 32
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)


print(df.isnull().sum())


# In[33]:


df.describe()


# In[34]:


#counting of instances exist for each unique diagnosis (e.g., malignant or benign).
df['diagnosis'].value_counts()


# Transform categorical into numerical labels. Here diagnosis

# In[35]:


lb=LabelEncoder()
df.iloc[:,1]=lb.fit_transform(df.iloc[:,1].values)


# In[36]:


df.head(21)


# ## Visualisation
# 
# Calculate the number of rows and columns needed in the subplot grid. Ensuring a maximum of 6 columns per row.
# 
# ### Seaborn Histogram Plots
# 
# Generate a grid of subplots and use Seaborn to create histogram plots (distplots) for each column in the DataFrame. The histograms display the distribution of numerical values, with 20 bins and kernel density estimation. Each subplot is labeled with the corresponding column name, and the y-axis grid is displayed for clarity.

# In[14]:


# Determine the number of rows and columns in the subplot grid
num_columns = len(df.columns)
num_rows = (num_columns - 1) // 6 + 1
num_cols = min(num_columns, 6)

# Plot displots for each column using seaborn
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

for i, column in enumerate(df.columns):
    row, col = divmod(i, num_cols)
    sns.histplot(df[column], bins=20, kde=True, color='blue', ax=axes[row, col])
    axes[row, col].set_title(column)
    axes[row, col].grid(axis='y')

plt.show()


# ### Seaborn Box Plots:
# Generate a grid of subplots, each containing a Seaborn box plot for a specific column in the DataFrame. The box plots visualize the distribution of numerical values, highlighting key statistics such as quartiles and potential outliers

# In[16]:


# Determine the number of rows and columns in the subplot grid
num_columns = len(df.columns)
num_rows = (num_columns - 1) // 6 + 1
num_cols = min(num_columns, 6)

# Plot box plots for each column using seaborn
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

for i, column in enumerate(df.columns):
    row, col = divmod(i, num_cols)
    sns.boxplot(x=df[column], ax=axes[row, col], color='blue')
    axes[row, col].set_title(column)
    axes[row, col].grid(axis='y')

plt.show()


# ## Handling Outliers with Interquartile Range method
# 
# Using the select_dtypes method to identify numerical columns (integers and floats) in the DataFrame (df) that may contain outliers.
# 
# For each identified numerical column with outliers, calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR). Determine lower and upper limits for identifying outliers based on a threshold of 1.5 times the IQR.
# 
# Identify and replace outliers in each column by setting values outside the determined limits to the median value of the respective column. This approach aims to mitigate the impact of outliers on the dataset while preserving central tendencies.

# In[37]:


import numpy as np
import pandas as pd


# Identify numerical columns with outliers
outlier_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Iterate over each outlier column
for outlier_column in outlier_columns:
    # Calculate Q1, Q3, and IQR for the column
    Q1 = df[outlier_column].quantile(0.25)
    Q3 = df[outlier_column].quantile(0.75)
    IQR = Q3 - Q1

    # Identify the lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers based on the limits
    outliers = ((df[outlier_column] < lower_limit) | (df[outlier_column] > upper_limit))

    # Replace outliers with the median value
    df.loc[outliers, outlier_column] = df[outlier_column].median()


# ### Visualizing Distributions After Outlier Removal
# 
# #### Seaborn Histogram Plots
# 
# All outliers have been successfully removed from the dataset, as evidenced by the updated distributions displayed in these plots compared to previous visualizations.

# In[22]:


# Determine the number of rows and columns in the subplot grid
num_columns = len(df.columns)
num_rows = (num_columns - 1) // 6 + 1
num_cols = min(num_columns, 6)

# Plot displots for each column using seaborn
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

for i, column in enumerate(df.columns):
    row, col = divmod(i, num_cols)
    sns.histplot(df[column], bins=20, kde=True, color='blue', ax=axes[row, col])
    axes[row, col].set_title(column)
    axes[row, col].grid(axis='y')

plt.show()


# #### Seaborn Box Plots

# In[23]:


# Determine the number of rows and columns in the subplot grid
num_columns = len(df.columns)
num_rows = (num_columns - 1) // 6 + 1
num_cols = min(num_columns, 6)

# Plot box plots for each column using seaborn
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

for i, column in enumerate(df.columns):
    row, col = divmod(i, num_cols)
    sns.boxplot(x=df[column], ax=axes[row, col], color='blue')
    axes[row, col].set_title(column)
    axes[row, col].grid(axis='y')

plt.show()


# ## Feature Selection
# 
# #### Visualization for Diagnosis Patterns
# Generating Pairplot:
# Creating a pairplot using Seaborn (sns.pairplot) for a subset of columns (columns 1 to 9) from the DataFrame, with hue defined by the 'diagnosis' variable.
# 
# Understanding Diagnosis Patterns:
# Utilizing the pairplot to visually assess the relationships and patterns among selected features, differentiated by the diagnosis (malignant or benign).

# In[38]:


sns.pairplot(df.iloc[:,1:10],hue="diagnosis")


# #### Feature Extraction
# Extracting the feature matrix X by removing the target variable ('diagnosis') from the original DataFrame df.
# Defining the target variable vector y as the 'diagnosis' column.
# 
# #### Correlation Analysis
# Computing the correlation matrix for all features in the dataset.
# 
# Sorting the absolute correlations of each feature with the target variable ('diagnosis') in descending order.
# 
# Selecting relevant features by excluding the target variable and considering those with the highest absolute correlations.

# In[53]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

correlation_matrix = df.corr()
correlation_with_target = correlation_matrix['diagnosis'].abs().sort_values(ascending=False)
relevant_features_corr = correlation_with_target[1:]  # Exclude the target variable

selected_features = list(relevant_features_corr.index)

# Display the selected features
print("Selected Features:", selected_features)


# In[124]:


#checking the null values 
print(X.isnull().sum())


# In[125]:


#printing the columns 
df.columns


# #### Creating new features
# Introducing new features such as the 'mean_area_radius_ratio,' 'texture_smoothness_ratio,' 'log_compactness_mean,' 'area_texture_ratio,' and 'compactness_symmetry_ratio' based on meaningful combinations of existing features.
# 
# **Dataset Splitting**
# 
# Splitting the original dataset into training and testing sets (80% training, 20% testing) to evaluate the model's performance.
# 
# **Feature Scaling**
# 
# Standardizing the feature values using StandardScaler to ensure all features are on a similar scale. This step is crucial for certain machine learning algorithms that are sensitive to the scale of input features.

# In[127]:


column_names = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]

# Assume 'diagnosis' is the target variable
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Feature Engineering: Creating new features
X['mean_area_radius_ratio'] = X['area_mean'] / X['radius_mean']
X['texture_smoothness_ratio'] = X['texture_mean'] / X['smoothness_mean']
X['log_compactness_mean'] = np.log1p(X['compactness_mean'])
X['area_texture_ratio'] = X['area_mean'] / X['texture_mean']
X['compactness_symmetry_ratio'] = X['compactness_mean'] / X['symmetry_mean']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# ### Support Vector Machine (SVM) Model Training and Evaluation
# **SVM Model Creation**
# Creating a Support Vector Machine (SVM) model using the SVC (Support Vector Classification) class with a random state for reproducibility.
# Model Training:
# 
# **Fitting the Model**
# Training the SVM model using the scaled training data (X_train_scaled and y_train) to learn the underlying patterns in the data.
# Model Evaluation:
# 
# **Predictions and Accuracy**
# Making predictions on the scaled test set (X_test_scaled) and assessing the model's performance using the accuracy metric.
# Printing the accuracy score to quantify how well the SVM model generalizes to unseen data.

# In[128]:


# Create an SVM model
svm_model = SVC(random_state=42)

# Train the SVM model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Model Accuracy: {accuracy_svm}')


# #### SVM Model Classification Report
# Generating a comprehensive classification report using the classification_report function. 
# 
# **Metrics Included**
# Precision: Ability of the classifier not to label as positive a sample that is negative.
# 
# Recall: Ability of the classifier to find all the positive samples.
# F1-score: Harmonic mean of precision and recall, providing a balance between the two metrics.
# 
# Support: Number of actual occurrences of the class in the specified dataset.
# 

# In[131]:


print("Classification Report:")
print(classification_report(y_test, y_pred_svm))


# #### SVM Model Confusion Matrix
# 
# Computing and displaying the confusion matrix using the confusion_matrix function. The confusion matrix summarizes the model's predictions in terms of true positive, true negative, false positive, and false negative instances.

# In[132]:


# Confusion Matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:")
print(conf_matrix_svm)


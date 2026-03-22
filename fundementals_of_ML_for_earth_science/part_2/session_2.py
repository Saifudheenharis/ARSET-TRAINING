import os
import sys
import csv
import glob
import joblib
import datasets
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning imports
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier as cumlRF
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, KFold
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

# Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate 

# Geospatial related imports
from osgeo import gdalconst
from osgeo import gdal

from pprint import pprint

# Ignore a few warnings for cleaner output
warnings.filterwarnings('ignore')

# Define General Variables
# url of the dataset we will be using, this is a link to the Hugging Face repository
# of this tutorial
DATASET_URL = 'nasa-cisto-data-science-group/modis-lake-powell-toy-dataset'

# controls random seed for reproducibility
RANDOM_STATE = 42

# column name for label, in our case this will be a categorical value
LABEL_NAME = 'water'

# data type of the label, you would change this to something else if your
# problem was for example a regression problem of type np.float32
DATA_TYPE = np.int16

# columns not needed for training
colsToDrop = ['x_offset', 'y_offset', 'year', 'julian_day']

# columns used as features during training
v_names = ['sur_refl_b01_1','sur_refl_b02_1','sur_refl_b03_1',
           'sur_refl_b04_1','sur_refl_b05_1','sur_refl_b06_1',
           'sur_refl_b07_1','ndvi','ndwi1','ndwi2']

# Here we create an output directory to store any artifacts out of our EDA visualizations
os.makedirs('output', exist_ok=True)

# Data Loading
# In this section we will go ahead and load our data to analyze. We have extracted a tabular dataset from MODIS GeoTIFF files for the purpose of performing EDA
dataset = datasets.load_dataset(DATASET_URL, split='train')
df_pandas = pd.DataFrame(dataset)
#print(df_pandas.head())

# Data Cleaning
# In this section we will start to inspect and understand the nature of our dataset
#df_pandas.info()

#df = df_pandas.describe().T
#print(tabulate(df, headers='keys', tablefmt='psql'))

# Checking for null values in the duplicate dataset
#df_pandas_test = df_pandas.copy()
#df_pandas_test.loc[0] = [np.nan, 1209, 1577, 743, 1028, 1969, 1932, 1587, 87, -2336, -914]
#df = df_pandas_test.describe().T

#Adding null value
#df = df_pandas_test[df_pandas_test.isnull().any(axis=1)]
#print(tabulate(df, headers='keys', tablefmt='psql'))

# Get a sample so we can speed up expensive visualizations
#sampledDf = df_pandas.sample(frac=0.1)
#sampledDf.info()

# EDA (Exploratory Data Analysis)
# Correlation plots with water points as orange
"""sns.set()
sns.pairplot(df_pandas, hue='water', palette='Set1',kind='reg')
plt.savefig('output/modisWaterTrainingEDA_Correlation_WaterHighlight.png')"""

# Distribution for each channel
colms = df_pandas.select_dtypes(include=['number']).columns
for col in colms:
    plt.figure(figsize=(8,4))
    sns.histplot(df_pandas[col], kde=True, color='teal')
    plt.title(f'Distribution of {col}')
    plt.show()





import os
import sys
import csv
import time
import glob
import joblib
import datasets
import datetime
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from pprint import pprint

from huggingface_hub import snapshot_download

from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.inspection import permutation_importance

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Geospatial related imports
from osgeo import gdalconst
from osgeo import gdal
import folium
from folium import plugins
import folium_helper

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# Define General Variables

# directory where we will output figures
FIGURE_OUTPUT_DIR = 'output2'

# directory where we will output raster
RASTER_OUTPUT_DIR = 'output2'

# directory where we will output our models
MODEL_OUTPUT_DIR = 'models2'

# url of the dataset we will be using, this is a link to the Hugging Face repository
# of this tutorial
DATASET_URL = 'nasa-cisto-data-science-group/modis-lake-powell-toy-dataset'

# directory where the raster data is downloaded
RASTER_DIR = 'raster'

# ratio of the dataset split for testing
TEST_RATIO = 0.2

# controls random seed for reproducibility
RANDOM_STATE = 42

# column name for label, in our case this will be a categorical value
LABEL_NAME = 'water'

# data type of the label, you would change this to something else if your
# problem was for example a regression problem of type np.float32
DATA_TYPE = np.int16

# Columns that are offset, years, julian days, etc (always need to be dropped).
offsets_indexes = ['x_offset', 'y_offset', 'year', 'julian_day', 'tileID']

# columns not needed for training
colsToDrop = ['x_offset', 'y_offset', 'year', 'julian_day']
colsToDropTraining = colsToDrop.copy()
colsToDropTraining.extend(offsets_indexes)

# columns used as features during training
v_names = ['sur_refl_b01_1','sur_refl_b02_1','sur_refl_b03_1',
           'sur_refl_b04_1','sur_refl_b05_1','sur_refl_b06_1',
           'sur_refl_b07_1','ndvi','ndwi1','ndwi2']

# Here we create an output directory to store any artifacts out of our models and visualizations
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# Data loading
train_dataset = pd.DataFrame(datasets.load_dataset(DATASET_URL, split='train'))
test_dataset = pd.DataFrame(datasets.load_dataset(DATASET_URL, split='test'))

# Splitting data from Hugging Face
"""X_train, y_train = train_dataset.drop(['water'], axis=1), train_dataset['water']
X_test, y_test = test_dataset.drop(['water'], axis=1), test_dataset['water']
print(X_train.shape, X_test.shape)"""

# Scikit splitting method
X_train, X_test, y_train, y_test = train_test_split(
    train_dataset.drop(['water'], axis='columns'),
    train_dataset['water'],
    random_state=RANDOM_STATE,
    stratify=train_dataset['water'],
    train_size=0.70,
)

# Training Preparation
kf = KFold(n_splits=5)

# Model fitting and training 
hyperparameters = {'n_estimators': 400, 
                   'criterion':'gini', 
                   'max_depth':None, 
                   'min_samples_split':2, 
                   'min_samples_leaf':1, 
                   'min_weight_fraction_leaf':0.0, 
                   'max_features':'sqrt', 
                   'max_leaf_nodes':None, 
                   'min_impurity_decrease':0.0, 
                   'bootstrap':True, 
                   'oob_score':False, 
                   'n_jobs':-1, 
                   'random_state':42, 
                   'verbose':0, 
                   'warm_start':True, 
                   'class_weight':None, 
                   'ccp_alpha':0.0, 
                   'max_samples':None
                  }

classifier = skRF(**hyperparameters)

# K-fold fitting
bestModel = None
bestModelScore = 0
scores = []
for trainIdx, testIdx in kf.split(X_train):
    #print("Train {}, Test {}".format(trainIdx, testIdx))
    X_train_valid, X_test_valid = X_train.iloc[trainIdx], X_train.iloc[testIdx]
    y_train_valid, y_test_valid = y_train.iloc[trainIdx], y_train.iloc[testIdx]
    #print('---------------------------------------')
    #print('Fitting model')
    st = time.time()
    classifier.fit(X_train_valid, y_train_valid)
    et = time.time()
    #print('Time to fit model: {}s'.format(et-st))
    #print('Getting score')
    score = classifier.score(X_test_valid, y_test_valid)
    if score>=bestModelScore:
        bestModelScore = score
        #print('Training accuracy score: {}'.format(score))
        bestModel = classifier
    #print('Predicting for test set')
    test_predictions = classifier.predict(X_test_valid)
    #print(classification_report(y_test_valid, test_predictions))
    #print('Score: {}'.format(score))
    scores.append(score)
    del test_predictions, score

# Average score and best score
scoreAvg = np.asarray(scores).mean()
#print('Average accuracy score: {}'.format(scoreAvg))
#print('Best accuracy score: {}'.format(bestModelScore))

# Regular fitting
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Average accuracy score for regular fitting: {}'.format(score))
print('Average accuracy score for K-fold fitting: {}'.format(bestModelScore))

# Model testing and data validation
classifier = bestModel
train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)
prediction_probs = classifier.predict_proba(X_test)

# Taking the only target values
predictionProbabilityList = list()
for i, subarr in enumerate(prediction_probs):
    predictionProbabilityList.append(subarr[1])
predictionProbabilityArray = np.asarray(predictionProbabilityList)

# Visualization
"""sns.displot(predictionProbabilityArray, bins=30)
plt.title('Distribution of the probability of predicted values')
plt.tight_layout()
plt.show()"""

# Altering data types
test_predictions = test_predictions.astype(np.int32)
y_test_int = y_test.astype(np.int32)

# Additional metrics
"""print('Test Performance')
print('-------------------------------------------------------')
print(classification_report(y_test, test_predictions))
cm = confusion_matrix(y_test_int, test_predictions)
recall = (cm[0][0] / (cm[0][0] + cm[0][1]))
print('Test Recall')
print('-------------------------------------------------------')
print(recall)
print('Confusion Matrix')
print('-------------------------------------------------------')
print(cm)"""


# Receiver Operating Characteristic (ROC) plots
clf = classifier

"""probs = clf.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'red', label = f'ROC AUC score = {roc_auc:.2f}')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()"""

# Restarted on 28/07/2026
# Permutation Importance
permutation_importance_results = permutation_importance(classifier,
                                                        X=X_test,
                                                        y=y_test,
                                                        n_repeats=10,
                                                        random_state=42)

png_save_path = 'mw_{}_{}_rf_{}_permutation_importance.png'.format(
    round(score, 3),
    hyperparameters['n_estimators'],
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

png_save_path = os.path.join(FIGURE_OUTPUT_DIR, png_save_path)

# Marker 28/07/2026
sorted_idx = permutation_importance_results.importances_mean.argsort()
plt.figure(figsize=(8, 8))
plt.barh(X_test.columns[sorted_idx], permutation_importance_results.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig(png_save_path)

del X_train, X_test, y_train, y_test, test_predictions, train_predictions, y_test_int

# Saving the model for future use
model_save_path = 'mw_{}_{}_{}_2.0.0_tuned_{}.sav'.format(
                                                          round(score, 3),
                                                          hyperparameters['n_estimators'],
                                                          'cpu',
                                                          datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
model_save_path = os.path.join(MODEL_OUTPUT_DIR, model_save_path)

print('Saving model to: {}'.format(model_save_path))
print(classifier)
joblib.dump(classifier, model_save_path, compress=3)

# Raster inference
# Data downloading
if not os.path.exists(os.path.join(RASTER_DIR, "e22fb0ce2c73d603ff182183fbfc1476d0032d1d")):
    print("Snapshot not found in this device, Starting download...")
    powell_dataset = snapshot_download(repo_id=DATASET_URL, allow_patterns="*.tif", repo_type='dataset')
    if os.path.exists(powell_dataset):
        shutil.copytree(os.path.dirname(powell_dataset), RASTER_DIR, dirs_exist_ok=True)
    else:
        print("Dataset not found in the cache")
else:
    powell_dataset = os.path.join(RASTER_DIR, "e22fb0ce2c73d603ff182183fbfc1476d0032d1d")

fileList = sorted([file for file in glob.glob(os.path.join(powell_dataset, 'IL.*.Powell.*.tif')) if 'sur_refl' in file])

# Feature engineering
def readRastersToArray(fileList):
    rasterProjection = None
    newshp = (1300*1300, 10)
    img = np.empty(newshp, dtype=np.int16)
    for i, fileName in enumerate(fileList):
        ds = gdal.Open(fileName)
        img[:, i] = ds.GetRasterBand(1).ReadAsArray().astype(np.int16).ravel()
        if i == 0:
            rasterProjection = ds.GetProjection()
            rasterTransform = ds.GetGeoTransform()
        ds = None
    img[:, len(fileList)] = ((img[:, 1] - img[:, 0]) / (img[:, 1] + img[:, 0])) * 10000
    img[:, len(fileList)+1] = ((img[:, 1] - img[:, 5]) / (img[:, 1] + img[:, 5])) * 10000
    img[:, len(fileList)+2] = ((img[:, 1] - img[:, 6]) / (img[:, 1] + img[:, 6])) * 10000
    return img, rasterProjection, rasterTransform

im, rasterProjection, rasterTransform = readRastersToArray(fileList)
"""print('Raster as ndarray')
print(im)
print('{} MB size'.format((im.size * im.itemsize) / 1000000))"""

# Data prepration
raster_dataframe = pd.DataFrame(im, columns=v_names, dtype=np.float32)
#print(raster_dataframe.describe())

# Prediction phase
def predictRaster(dataframe, colsToDrop=None):
    """
    Function given a raster in the form of a 
    GPU/CPU-bound data frame then perform 
    predictions given the loaded model.
    
    Return the prediction matrix, the prediction probabilities
    for each and the dataframe converted to host.
    """
    df = dataframe.drop(columns=colsToDrop) if colsToDrop else dataframe
    print('Making predictions from raster')
    predictions = classifier.predict(df).astype(np.int16)
    predictionsProbs = classifier.predict_proba(df).astype(np.float32)
    return predictions, predictionsProbs

predictedRaster, predictedProbaRaster = predictRaster(raster_dataframe)
raster_shape = (1300, 1300)
predictedRasterNdArray = np.asarray(predictedRaster)
predictedRasterMatrix = predictedRasterNdArray.reshape(raster_shape)
#print(predictedRasterMatrix)

# Postprocessing (QA)
qa = [file for file in glob.glob(os.path.join(powell_dataset, 'IL.*.Powell.*.tif')) if 'qa' in file][0]
ds = gdal.Open(qa)
qaMask = ds.GetRasterBand(1).ReadAsArray()
raster_qad = np.where(qaMask == 0, predictedRasterMatrix, 255)

# Visualizing the QA + Prediction mask
"""plt.matshow(raster_qad)
plt.colorbar()"""

# Creating a new GeoTiff file for the mask

rasterTransform
predictedPath = 'PowellPredictedWaterMask.tif'

driver = gdal.GetDriverByName('GTiff')
outDs = driver.Create(predictedPath, 1300, 1300, 1, gdal.GDT_Int16, options=['COMPRESS=LZW'])
outDs.SetGeoTransform(rasterTransform)
outDs.SetProjection(rasterProjection)
outBand = outDs.GetRasterBand(1)
outBand.WriteArray(raster_qad)
outBand.SetNoDataValue(255)
outDs.FlushCache()
outDs = None
outBand = None
driver = None



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
import xarray as xr
import rioxarray as rxr
from rioxarray import merge
from pathlib import Path
from pprint import pprint

from huggingface_hub import snapshot_download

from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.inspection import permutation_importance

# import useful packages
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import natsort

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
#%matplotlib inline

# GPU libraries
import cudf
import cuml
import cupy as cp
from cuml.cluster import KMeans as KMeansGPU

DATASET_URL = 'nasa-cisto-data-science-group/modis-lake-powell-toy-dataset'

powell_dataset = snapshot_download(repo_id=DATASET_URL, allow_patterns="*.tif", repo_type='dataset')

fileList = sorted([file for file in glob.glob(os.path.join(powell_dataset, 'IL.*.Powell.*.tif')) if 'sur_refl' in file])

raster_arrays = []
for val in fileList:
    raster_arrays.append(rxr.open_rasterio(val))

raster = xr.concat(raster_arrays, dim="band")
raster

(raster[:3, :, :] / 10000).plot.imshow()

# grab the number of bands in the image, naip images have four bands
nbands = raster.shape[0]

# create an empty array in which each column will hold a flattened band
flat_data = np.empty((raster.shape[1]*raster.shape[2], nbands))

# loop through each band in the image and add to the data array
for i in range(nbands):
    band = raster[i,:,:].values
    flat_data[:, i-1] = band.flatten()


print(flat_data, type(flat_data))

#Converting to GPU format

flat_data_gpu = cp.asarray(flat_data)
flat_data_gpu, type(flat_data_gpu)

#Initalizing K-means GPU mode
# set up the kmeans classification by specifying the number of clusters 
km = KMeansGPU(n_clusters=N_CLUSTERS)

# begin iteratively computing the position of the two clusters
km.fit(flat_data_gpu)

# use the sklearn kmeans .predict method to assign all the pixels of an image to a unique cluster
flat_predictions = cp.asnumpy(km.predict(flat_data_gpu))

# rehsape the flattened precition array into an MxN prediction mask
prediction_mask = flat_predictions.reshape((raster.shape[1], raster.shape[2]))

# plot the imagery and the prediction mask for comparison
f, axarr = plt.subplots(1,2)
axarr[0].imshow(np.moveaxis(raster[:3,:,:].values, 0, -1) / 10000)
axarr[0].set_title('Imagery')
axarr[1].imshow(prediction_mask)
axarr[1].set_title('kmeans predictions')

axarr[0].axis('off')
axarr[1].axis('off')

plt.show()


import os
import sys
import csv
import time
import glob
#import joblib
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

DATASET_URL = 'nasa-cisto-data-science-group/modis-lake-powell-toy-dataset'
powell_dataset = snapshot_download(repo_id=DATASET_URL,local_dir="/home/saif/arset",local_dir_use_symlinks="auto", allow_patterns="*.tif", repo_type='dataset')

fileList = sorted([file for file in glob.glob(os.path.join(powell_dataset, 'IL.*.Powell.*.tif')) if 'sur_refl' in file])

raster_arrays = []
for val in fileList:
    raster = rxr.open_rasterio(val)
    raster_arrays.append(raster)
    #(raster.sel(band=1)/10000).plot.imshow()
    #plt.show()

raster = xr.concat(raster_arrays, dim="band")
#(raster[:3, :, :] / 10000).plot.imshow()
#plt.show()

#K-Means Clustering
# number of clusters
N_CLUSTERS = 4

# grab the number of bands in the image, naip images have four bands
nbands = raster.shape[0]

# create an empty array in which each column will hold a flattened band
flat_data = np.empty((raster.shape[1]*raster.shape[2], nbands))

# loop through each band in the image and add to the data array
#Marker 24/01/2026
for i in range(nbands):
    band = raster[i,:,:].values
    flat_data[:, i-1] = band.flatten()
    
# set up the kmeans classification by specifying the number of clusters 
km = KMeans(n_clusters=N_CLUSTERS)

# begin iteratively computing the position of the two clusters
km.fit(flat_data)

# use the sklearn kmeans .predict method to assign all the pixels of an image to a unique cluster
flat_predictions = km.predict(flat_data)

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

fig, axes = plt.subplots(1,4, figsize=(12, 12))
for n, ax in enumerate(axes.flatten()):
    ax.imshow(prediction_mask==[n], cmap='gray');
    ax.set_axis_off()
    
fig.tight_layout()
plt.show()

#! /usr/bin/env python3
# coding: utf-8

from pyspark.sql import SparkSession

# Create Spark session
spark = (SparkSession.builder
                     .appName('DS_P8')
                     .config("spark.sql.parquet.writeLegacyFormat", ('true'))
                     .getOrCreate()
        )
sc = spark.sparkContext
# avoid the "_success" file when saving data
sc._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

import os
import io
from typing import Iterator
import numpy as np
import pandas as pd
from PIL import Image
import scipy
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from pyspark.sql.functions import regexp_extract, pandas_udf, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml import Pipeline

# Define paths
path = "s3://per-oc-project8/data/"
data_path = os.path.join(path, "Test")
result_path = os.path.join(path, "result_parquet")

def load_img(dir_path):
    """
    Load all .jpg images saved in a directory to a binary Spark DataFrame.
    """
    images = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(dir_path)
    return images

def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    model.set_weights(bc_model_weights.value)
    return model

def preprocess_img(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, series):
    """
    Featurize a pd.Series of raw images using the MobileNetV2 model.
    For some layers, output features will be multi-dimensional tensors.
    Feature tensors are flattened to vectors for easier storage in Spark DataFrames.
    -----------    
    Return: a pd.Series of image features
    """
    X = np.stack(series.map(preprocess_img))
    preds = model.predict(X)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>')
def featurize_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column
    of type ArrayType(FloatType).
    With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    for multiple data batches.
    This amortizes the overhead of loading big models.
    """
    model = model_fn()
    for s in batch_iter:
        yield featurize_series(model, s)

def dim_red_series(model, series):
    """
    Reduce the dimension of a pd.Series of features using Sparse Random Projection.
    -----------    
    Return: a pd.Series of image features
    """
    X = np.stack(series)
    X_tr = model.transform(X)
    return pd.Series(X_tr.tolist())

@pandas_udf('array<float>')
def reduce_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """This method is a Scalar Iterator pandas UDF
    wrapping our dimensionality reduction function.
    The decorator specifies that this returns a Spark DataFrame column
    of type ArrayType(FloatType).
    With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    for multiple data batches.
    This amortizes the overhead of loading big models.
    """
    model = bc_srp.value
    for s in batch_iter:
        yield dim_red_series(model, s)
  
if __name__ == '__main__':
    # Load images
    df_img = load_img(data_path).coalesce(16)
    n_img = df_img.count()
    print(f"number of images: {n_img}")
    # Get labels
    regex = r'(.*)/(.*[a-zA-Z])(.*)/'
    df_img = df_img.withColumn('label', regexp_extract('path', regex, 2))
    # Get image features via transfert learning
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    bc_model_weights = sc.broadcast(model.get_weights())
    df_features = df_img.select(
        'label',
        featurize_udf('content').alias('features')
    )
    # Create a sparse dummy array to fit the Sparse Random Projection
    # It enables to broadcast the fitted sparse random projection
    n_features = len(df_features.first()['features'])
    dummy_X = scipy.sparse.csr_matrix((n_img, n_features), dtype=np.float32)
    # Create the Sparse Random Projection model
    k = johnson_lindenstrauss_min_dim(n_features, eps=0.1)
    srp = SparseRandomProjection(n_components=k, random_state=42)
    srp.fit(dummy_X)
    bc_srp = sc.broadcast(srp)
    # Reduce feature dimension with the Sparse Random Projection model
    result = df_features.select(
        'label',
        reduce_udf('features').alias('features')
    )
    # Save results
    result.coalesce(1).write.mode("overwrite").parquet(result_path)
    print("Data have been successfully saved")

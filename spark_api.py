#! /usr/bin/env python3
# coding: utf-8

# Standard libraries
import os
import io
from typing import Iterator
# Import numpy and pandas for data manipulation
import numpy as np
import pandas as pd
# image preprocessing
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# Import deep learning model with tensorflow
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# Import pyspark library
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, pandas_udf, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml import Pipeline

# Create Spark session
spark = (SparkSession.builder
                     .appName('DS_P8')
                     .config("spark.sql.parquet.writeLegacyFormat", ('true'))
                     .getOrCreate()
        )
sc = spark.sparkContext

# Define variables
path = 's3://per-oc-project8/data/'
data_path = os.path.join(path, 'local_test')
output_path = os.path.join(path, 'output_parquet')
model = MobileNetV2(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                   )
bc_model_weights = sc.broadcast(model.get_weights())

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
    model = MobileNetV2(weights=None,
                        include_top=False,
                        input_shape=(224, 224, 3))
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
    input = np.stack(series.map(preprocess_img))
    preds = model.predict(input)
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

def main():
    # Load images
    df_img = load_img(data_path)
    n_img = df_img.count()
    # Get labels
    regex = r'(.*)/(.*[a-zA-Z])(.*)/'
    df_img = df_img.withColumn('label', regexp_extract('path', regex, 2))
     # Get image features via transfert learning
    df_features = df_img.select(
        'label',
        featurize_udf('content').alias('features')
    )
    # Transform features to a vector type
    arr_to_vec_udf = udf(lambda a: Vectors.dense(a), VectorUDT())
    df_vec = df_features.select(
        'label',
        arr_to_vec_udf('features').alias('features')
    )
    df_vec.show(5)
    # Scale data
    scaler = StandardScaler(
        withMean=True,
        withStd=False,
        inputCol='features',
        outputCol='scaled_features'
    )
    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec).select(['label', 'scaled_features'])
    df_scaled.show(5)
    # # Run PCA with all components
    # n_features = 7 * 7 * 1280
    # n = min(n_img, n_features)
    # pca = PCA(k=n, inputCol='features', outputCol='pca_features')
    # pca_model = pca.fit(df_scaled)
    # # Get the right number of components to keep 95% of the variance
    # cumsum = np.cumsum(pca_model.explainedVariance)
    # d = np.argmax(cumsum >= 0.95) + 1
    # print(f"Number of principal components to keep : {d}")
    # pca_model.setK(d)
    # result = pca_model.transform(df_scaled).select('label', 'pca_features')
    # Save results
    df_scaled.write.mode("overwrite").parquet(output_path)   

if __name__ == '__main__':
    main()

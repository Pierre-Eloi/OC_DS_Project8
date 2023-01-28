aws emr add-steps \
    --cluster-id $1 \
    --steps Type=SPARK,Name="P8_Fruits",Args=[--deploy-mode,cluster,s3://per-oc-project8/spark_api.py]
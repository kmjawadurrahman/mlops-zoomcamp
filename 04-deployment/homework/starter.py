#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


# output_file = "predictions.parquet"


def read_data(filename, categorical_columns_list):
    df = pd.read_parquet(filename)

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical_columns_list] = (
        df[categorical_columns_list].fillna(-1).astype("int").astype("str")
    )

    return df


def run_prediction(year, month):
    categorical = ["PUlocationID", "DOlocationID"]
    print("reading data...")
    df = read_data(
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet",
        categorical,
    )
    print("loading model...")
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"prediction mean: {y_pred.mean()}")

    # df_result = pd.DataFrame({"predictions": y_pred})

    # df_result["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    run_prediction(year, month)

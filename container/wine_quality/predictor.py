import io
import json
import os

import joblib
import pandas as pd
from flask import Flask, Response, request

MODEL_FILENAME = "model.pkl"
MODEL_PATH = os.path.join("/opt/ml/model", MODEL_FILENAME)

app = Flask(__name__)
model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/ping", methods=["GET"])
def ping():
    try:
        load_model()
        return Response(response="\n", status=200, mimetype="application/json")
    except Exception:
        return Response(response="\n", status=404, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    # Expect CSV payload with 11 columns and no header.
    payload = request.data.decode("utf-8")
    df = pd.read_csv(io.StringIO(payload), header=None)
    preds = load_model().predict(df)
    return Response(response=json.dumps(preds.tolist()), status=200, mimetype="application/json")

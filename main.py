from flask import Flask, jsonify, request
from prediction import predict

app = Flask(__name__)


@app.route('/')
def home():
    return "<h1>Logistics Prediction<h1>"


@app.route('/api/predict')
def out_predict():
    input_data = request.args.get('predict')
    response = predict(input_data)

    return jsonify(response)


app.run()

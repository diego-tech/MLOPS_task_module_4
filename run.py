from flask import Flask, request
import os
from app.src.models import train_model
from app.src.models.predict import predict_pipeline
from app import ROOT_DIR
import warnings

# Remove Warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Set PORT
port = int(os.getenv('PORT', 8080))

# Main Route
@app.route('/', methods=['GET'])
def root():
    return {'Proyecto': 'Mod. 4 - Ciclo de vida de modelos IA'}

@app.route('/train-model', methods=["GET"])
def train_mode_route():
    # Data Path
    data_path = os.path.join(ROOT_DIR, "data/data.csv")

    train_model.training_pipeline(data_path)

    return {"Training Model": "Mod. 4 - Ciclo de vida de modelos IA"}

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()

    y_pred = predict_pipeline(data)

    return {'Predicted value': y_pred}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
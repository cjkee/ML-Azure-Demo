from flask import Flask, request, jsonify
import pickle

# Create Flask app
app = Flask(__name__)

# Load the machine learning model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Welcome to the ML Model API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()

        # Assuming the model expects a list of features
        features = data.get("features")
        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Make prediction
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

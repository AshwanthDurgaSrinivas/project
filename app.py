from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model
tokenizer = T5Tokenizer.from_pretrained("srinu590/project")
model = T5ForConditionalGeneration.from_pretrained("srinu590/project")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    input_text = ' '.join(symptoms)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"prediction": output_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

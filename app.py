from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)
CORS(app)

tokenizer = T5Tokenizer.from_pretrained("srinu590/t5-small")
model = T5ForConditionalGeneration.from_pretrained("srinu590/t5-small")
device = torch.device("cpu")
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        symptoms = data.get("symptoms", [])

        if not symptoms or not isinstance(symptoms, list):
            return jsonify({"error": "Invalid symptoms format"}), 400

        input_text = ' '.join(symptoms)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"prediction": output_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

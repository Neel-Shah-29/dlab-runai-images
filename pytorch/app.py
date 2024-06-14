from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the tokenizer and model
model_name = 'meta-llama/Meta-Llama-3-8B'  # Adjust this path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(jsonify({"response": response}))
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
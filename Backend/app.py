from flask import Flask, request, render_template_string, jsonify
from Model.model import ModelLLM
from website import HTML_PAGE
from Model.extractive import generate_title as generate_title

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE), 200

@app.route('/generate_title', methods=['POST'])
def summarize():
    data = request.get_json()
    if 'model' in data:
        if data['model'] != 'extractive':
            model = ModelLLM(data['model'])
        else:
            summary = generate_title(data['text'])
            return jsonify({'title': summary}), 200
    else:
        model = ModelLLM("facebook/bart-large-cnn")
    text = data['text']
    summary = model.infer(text)
    return jsonify({"title": summary}), 200

if __name__ == '__main__':
    app.run(port=33345, debug=True)
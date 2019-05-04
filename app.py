from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def get():
    return jsonify({'msg': 'Welcome to Gender Predictor Api'})


@app.route('/predict', methods=['POST'])
def predict_gender():
    name = request.json['name']
    # TODO:: prediction task here
    return jsonify({'name': name, 'gender': "Unknown"})


if __name__ == '__main__':
    app.run(debug=True)

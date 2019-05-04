from flask import Flask, request, jsonify
from predictor import GenderPredictor, names_data

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get():
    return jsonify({'msg': 'Welcome to Gender Predictor Api'})


@app.route('/predict', methods=['POST'])
def predict_gender():
    name = request.json['name']
    gender = genderPredictor.predict(name.strip().lower())
    return jsonify({'name': name, 'gender': gender})


if __name__ == '__main__':
    genderPredictor = GenderPredictor(names_data)
    genderPredictor.trainClassifier()
    genderPredictor.testAccuracy()
    app.run(debug=True)


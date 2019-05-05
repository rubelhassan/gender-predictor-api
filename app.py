from flask import Flask, request, jsonify
from predictor import GenderPredictor, trainFeatures, ClassifierType

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
    genderPredictor = GenderPredictor(trainFeatures, ClassifierType.RANDOM_FOREST)
    genderPredictor.trainClassifier()
    genderPredictor.accuracy()
    app.run(debug=True)


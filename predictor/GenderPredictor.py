import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


class GenderPredictor:
    le = preprocessing.LabelEncoder()

    def __init__(self, names_data):
        self.names_array = np.array(names_data)
        self.classifer = GaussianNB()

    def trainClassifier(self):
        self.processTrainingData()
        self.fitClassifer()

    def getNamesData(self):
        return self.names_data

    def processTrainingData(self):
        nick_names = [f.strip().lower() for f in self.names_array[:, 0]]
        names_encoded = [[data] for data in self.le.fit_transform(nick_names)]
        endsWithVowel = [[self.doesEndWithVowel(name)]
                         for name in nick_names]
        endsWithSonrants = [[self.doesEndWithSonarant(name)]
                            for name in nick_names]
        namesLengths = [[len(name)] for name in nick_names]
        self.names = np.concatenate((names_encoded, endsWithVowel, endsWithSonrants, namesLengths), axis=1)
        self.genders = self.names_array[:, 1]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.names, self.genders, test_size=0.3, random_state=109)

    def fitClassifer(self):
        self.classifer.fit(self.X_train, self.y_train)

    def testAccuracy(self):
        y_pred = self.classifer.predict(self.X_test)
        print("Accuracy: {}".format(metrics.accuracy_score(self.y_test, y_pred)))

    def predict(self, name):
        name_encoded = self.le.fit_transform([name])
        name_features = np.concatenate(([name_encoded], [[self.doesEndWithVowel(name)]]), axis=1)
        predictedGender = self.classifer.predict(name_features)
        return 'Male' if 'm' == predictedGender[0] else 'Female'

    def doesEndWithVowel(self, name):
        return name.endswith(('a', 'e', 'i', 'o', 'u'))

    def doesEndWithSonarant(self, name):
        return name.endswith(('m', 'n', 'l', 'w', 'j', 'r'))

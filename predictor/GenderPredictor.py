import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from .ClassifierType import ClassifierType


class GenderPredictor:
    le = preprocessing.LabelEncoder()

    def __init__(self, train_features, classifier_type):
        self.namesFeatures = np.array(train_features)
        self.classifier = self.determineClassifier(classifier_type)

    def trainClassifier(self):
        self.prepareTrainingData()
        self.fitClassifer()

    def getNamesData(self):
        return self.namesFeatures

    def prepareTrainingData(self):
        nickNames = [f.strip().lower() for f in self.namesFeatures[:, 0]]
        namesEncoded = [[data] for data in self.le.fit_transform(nickNames)]
        endsWithVowel = [[self.endsWithVowel(name)] for name in nickNames]
        endsWithSonrants = [[self.endsWithSonorant(name)] for name in nickNames]
        namesLengths = [[len(name)] for name in nickNames]
        self.names = np.concatenate((namesEncoded, endsWithVowel, endsWithSonrants, namesLengths), axis=1)
        self.genders = self.namesFeatures[:, 1]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.names, self.genders, test_size=0.3, random_state=44, shuffle=True)

    def fitClassifer(self):
        self.classifier.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.classifier.predict(self.X_test)
        print("Accuracy: {}".format(metrics.accuracy_score(self.y_test, y_pred)))

    def predict(self, name):
        name_encoded = self.le.fit_transform([name])
        name_features = np.concatenate(([name_encoded],
                                [[self.endsWithVowel(name)]], [[self.endsWithSonorant(name)]], [[len(name)]]), axis=1)
        predictedGender = self.classifier.predict(name_features)
        return 'Male' if 'm' == predictedGender[0] else 'Female'

    @staticmethod
    def endsWithVowel(name):
        return name.endswith(('a', 'e', 'i', 'o', 'u'))

    @staticmethod
    def endsWithSonorant(name):
        return name.endswith(('m', 'n', 'l', 'w', 'j', 'r'))

    def determineClassifier(self, classifier_type):
        if not isinstance(classifier_type, ClassifierType):
            raise TypeError('No classifier type provided for GenderPredictor')

        if classifier_type == ClassifierType.G_NAIVE:
            return GaussianNB()

        return RandomForestClassifier(n_jobs=-1)

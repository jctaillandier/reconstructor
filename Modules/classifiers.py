# spot check machine learning algorithms on the adult imbalanced dataset
from numpy import mean
from numpy import std
import pdb, argparse
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BaseClassifiers:
    def __init__(self, file_to_test, path_to_exp, kfold='false'):
        self.file_name1 = file_to_test
        self.path = path_to_exp
        self.kfold = kfold


    def load_dataset(self, full_path):
        # load the dataset as a numpy array
        # dataframe = read_csv(full_path, header=None, na_values='?')
        dataframe = read_csv(full_path,na_values='?')
        # drop rows with missing
        dataframe = dataframe.dropna()
        # split into inputs and outputs
        last_ix = 'sex'
        X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]

        # select categorical and numerical features
        cat_ix = X.select_dtypes(include=['object', 'bool']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        c_ix = []
        n_ix = []
        for i, v in enumerate(dataframe.columns.tolist()):
            if v in cat_ix:
                c_ix.append(i)
            elif v in num_ix:
                n_ix.append(i)
        # label encode the target variable to have the classes 0 and 1
    #     y = pd.to_numeric(y.values[:,0])
        # y = LabelEncoder().fit_transform(y)
        return X.values, y.values, c_ix, n_ix


    # evaluate a model
    def evaluate_model(self, X, y, model, kfold):
        if kfold.lower() == 'false':
            perc=0.15
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test,preds)
        
        else:
            k = 10
            # define evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=1)
            
            # evaluate model
            score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=2)

        return score

    # define models to test
    def get_models(self):
        models, names = list(), list()
        # CART
        # models.append(DecisionTreeClassifier())
        # names.append('CART')
        # SVM
        models.append(SVC(gamma='scale'))
        names.append('SVM')
    #     # Bagging
        models.append(BaggingClassifier(n_estimators=100))
        names.append('BAG')
        # RF
        models.append(RandomForestClassifier(n_estimators=100))
        names.append('RF')
    # #     # GBM
        models.append(GradientBoostingClassifier(n_estimators=100))
        names.append('GBM')
        return models, names

    def runit(self):
        # file = 'train_sex'
        full_path = f'{self.path}models_data/{self.file_name1}.csv'
        # full_path = f"../GeneralDatasets/Csv/disp_impact_decoded2222.csv"
        X, y, cat_ix, num_ix = self.load_dataset(full_path)

        # define models
        models, names = self.get_models()
        results = list()
        total_texts = []
        # evaluate each model

        for i in range(len(models)):
            # define steps
            steps = [('c',OneHotEncoder(handle_unknown='ignore'),cat_ix), ('n',MinMaxScaler(),num_ix)]
            # one hot encode categorical, normalize numerical
            ct = ColumnTransformer(steps)
            # wrap the model i a pipeline
            pipeline = Pipeline(steps=[('t',ct),('m',models[i])])

            # evaluate the model and store results
            scores = self.evaluate_model(X,y, pipeline, self.kfold)
            results.append(scores)
            # summarize performance
            print_text = "{},{:.2f},{:.4f}".format(names[i], mean(scores), std(scores))
            
            total_texts.append(print_text)

        total_score = 0
        for result in total_texts:
            score = result.split(',')[1]
            total_score = total_score + float(score)
        average = total_score/len(total_texts)

        with open(f"{self.path}external_classif_{self.file_name1}.txt", 'w+') as f:
                    f.write(f"Input file: {self.file_name1}")
                    f.write(f"Results: {results} \n \n")
                    f.write(f"Test Output: {total_texts}\n")
                    f.write(f"Classifiers average:{average}")

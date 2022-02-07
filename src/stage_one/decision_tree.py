import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

def predict(data):
    predictions = []
    with open('./stage_one/dec_tree.pickle', 'rb') as f:
        clf = pickle.load(f)
        predictions = clf.predict(data)
    return predictions

def import_data(path):
    data = pd.read_csv(
        path,
        usecols=lambda col: col not in ['Convexity', 'Extend', 'Min', 'Geometric Mean', 'Sum']
    )
    data.rename(
        columns=({'Num Pix': 'Area'}),
        inplace=True
    )
    # Change all parasite/vacuole that are greater than 4 to be '>4'.
    # data.loc[data['Parasites/vacuole'] > 4, 'Parasites/vacuole'] = '>4'
    # Change all parasite/vacuole from int to string.
    # data['Parasites/vacuole'] = data['Parasites/vacuole'].map(str)
    return data

def build(path):
    # Build the decision tree
    data = import_data(path)
    # Values in the first 6 columns are the dependent variables
    X = data.values[:, 0:6]
    # Value in the last column is the one to be measured - it is the
    # pathogen/vacuole.
    Y = data.values[:, 6]
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.9, random_state = 100)
    
    clf_gini = DecisionTreeClassifier(criterion = "gini",
                                      max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    
    # Below is testing how well the model is in predicting the number of
    # pathogens per vacuole.
    y_pred = clf_gini.predict(X_test)
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))
    with open('./dec_tree.pickle', 'wb') as f:
        pickle.dump(clf_gini, f)

if __name__ == '__main__':
    build('../training_decision_tree/Training_dataset_Salmonella_t2hrs.csv')
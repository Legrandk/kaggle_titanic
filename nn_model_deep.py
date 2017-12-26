# -*- coding: utf-8 -*-

# Titanic Challenge: https://www.kaggle.com/c/titanic/data

"""
Simplest Model:
    Architecture: 
        Input:  29 features
        Output: 1 neuron
    
    >> CM: [[103  11]
            [ 12  53]]
    >> Dev ACC: 0.871508379888
    >> Dev PREC: 0.828125
    >> Dev RECALL: 0.815384615385
    >> Dev F1_Score: 0.821705426357


Model#9
    Architecture: 
        Input:  29 features
        Hidden: 5 neurons
        Output: 1 neuron

    >> Dev CM: [[104  10]
                [ 13  52]]    
    >> Train ACC: 0.839724208376
    >> Dev ACC: 0.871508379888
    >> Dev PREC: 0.838709677419
    >> Dev RECALL: 0.8
    >> Dev F1_Score: 0.818897637795


Model#8 got the best results:
    Position: 3,342 (Top 35%)
    Scoring:  0.78947

    Architecture: 
        Input:  29 features
        Hidden: 9 neurons
        Hidden: 9 neurons
        Hidden: 9 neurons
        Output: 1 neuron

    L2 Reg: 0.12
    Weights Init: glorot_normal
    Optimizer: Adam (LR: 0.001)
    Epochs: 400
    
    >> CM: [[103  11]
            [ 14  51]]
    >> Dev ACC: 0.860335195531
    >> Dev PREC: 0.822580645161
    >> Dev RECALL: 0.784615384615
    >> Dev F1_Score: 0.803149606299
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import regularizers


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#DESCOMENTAR ESTO EN KAGGLE!!!!!!!
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#TRAIN_CSV = "../input/train.csv"
#TEST_CSV  = "../input/test.csv"
TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"

OUTPUT_CSV = "survivors_prediction.csv"
VERBOSE    = 0
SEED       = 123

#########################################################

np.random.seed( SEED)


"""
Feature Engineering by Ahmed BESBES:
    https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
"""

def get_combined_data( train_filename, test_filename):
    train = pd.read_csv( train_filename)
    test  = pd.read_csv( test_filename)
    
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return targets, combined


def get_titles( combined):
    combined['Title'] = combined['Name'].map( lambda name: name.split(',')[1].split('.')[0].strip())

    # Officer, Royalty, Master, Mr, Mrs, Miss
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    combined['Title'] = combined.Title.map(Title_Dictionary)
    print("get_titles done")
    return combined
    
    
def process_age( combined):
    def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.median()

    grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
    grouped_median_test = grouped_test.median()
            
    combined.head(891).Age = combined.head(891).apply(
            lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    combined.iloc[891:].Age = combined.iloc[891:].apply(
            lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
    print("process_age done")
    return combined


def process_names( combined):
    combined.drop('Name', inplace=True, axis = 1)
    
    title_dummies = pd.get_dummies( combined['Title'], prefix='Title')
    combined = pd.concat( [combined, title_dummies], axis = 1)
    
    combined.drop('Title', inplace=True, axis = 1)
    
    print("process_names done")
    return combined
    
    
def process_fares( combined):
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)    
    print('process_fare done')
    return combined


def process_embarked( combined):
    combined.Embarked.fillna('U', inplace=True)
    
    embarked_dummies = pd.get_dummies( combined['Embarked'], prefix='Embark')
    combined = pd.concat([combined, embarked_dummies], axis = 1)
    
    combined.drop( 'Embarked', inplace=True, axis=1)
    print('process_embarked done')
    return combined
    

def process_cabin( combined):
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
    combined = pd.concat([combined,cabin_dummies], axis=1)
    
    combined.drop('Cabin', axis=1, inplace=True)
    print('process_cabin done')
    return combined
    
    
    
def process_sex( combined):
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    print('process_sex done')
    return combined
    
    
def process_pclass( combined):
    pclass_dummies = pd.get_dummies( combined['Pclass'], prefix='Pclass')
    combined = pd.concat([combined, pclass_dummies], axis=1)
    
    combined.drop('Pclass', inplace=True, axis=1)
    print('process_pclass done')
    return combined
    
    
def process_family( combined):
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    
    combined.drop('Parch', inplace=True, axis=1)
    combined.drop('SibSp', inplace=True, axis=1)
    
    print('process_family done')
    return combined


def preprocess_data( combined, features, drop_features):
    
    for feature in features:
        if 'titles' == feature:
            combined = get_titles( combined)
        elif 'age' == feature:
            combined = process_age( combined)
        elif 'names' == feature:
            combined = process_names( combined)
        elif 'fares' == feature:
            combined = process_fares( combined)
        elif 'embarked' == feature:
            combined = process_embarked( combined)
        elif 'cabin' == feature:
            combined = process_cabin( combined)
        elif 'sex' == feature:
            combined = process_sex( combined)
        elif 'pclass' == feature:
            combined = process_pclass( combined)
        elif 'family' == feature:
            combined = process_family( combined)
    
    
    for feature in drop_features:
        combined.drop( feature, inplace=True, axis=1)
        print ("Feature '{}' dropped".format(feature))
    
    
    return combined


def recover_train_test_target( combined):
    train = combined.head(891)
    test  = combined.iloc[891:]
    
    return train, test



def get_metrics( y, y_hat):
    cm = confusion_matrix( y, (y_hat > 0.5))

    TP = cm[1][1] 
    TN = cm[0][0] 

    FP = cm[0][1]
    FN = cm[1][0]

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / ( precision + recall)

    return cm, acc, precision, recall, f1_score


df_labels, combined = get_combined_data( TRAIN_CSV, TEST_CSV)
combined = preprocess_data( combined, 
                              ['titles', 
                               'age', 
                               'names', 
                               'fares', 
                               'embarked', 
                               'cabin', 
                               'sex', 
                               'pclass', 
                               'family'], 
                               [ 'PassengerId', 'Ticket'])


df_train, df_test = recover_train_test_target( combined)

print(">> Train number of samples: {}".format(df_train.shape[0]))
print(">> Test  number of samples: {}".format(df_test.shape[0]))


# Features matrix
X = df_train.as_matrix()
# Survivor Vector
y = df_labels.as_matrix()


# Splitting the dataset into two sets: Training, Dev and Test
X_train, X_dev, y_train, y_dev = train_test_split( X, y, test_size=0.2, random_state = SEED)

# Feature Standardization(mu=0, std=1)
sc_X    = StandardScaler()
X_train = sc_X.fit_transform(X_train)   #(534,853)  => (712,29)
X_dev   = sc_X.transform(X_dev)         #(179,853)  => (179,29)


print("X_train.shape: {}".format(X_train.shape))
print("X_dev.shape: {}".format(X_dev.shape))


#
# NN MODEL
#

model = Sequential()

model.add( Dense( units              = 5,
                  input_dim          = X_train.shape[1],
                  kernel_initializer = 'glorot_normal',
                  activation         = 'relu'))

model.add( Dense( units              = 1,
                  kernel_initializer = 'glorot_normal',
                  activation         = 'sigmoid'))

model.compile( optimizer = 'adam',
               loss = 'binary_crossentropy',
               metrics = [ 'accuracy' ])

history = model.fit( X_train , y_train,
           batch_size = 32,
           epochs = 275, #400,
           verbose = VERBOSE)

y_hat = model.predict( X_dev)
dev_cm, dev_acc, dev_precision, dev_recall, dev_f1_score = get_metrics( y_dev, y_hat)

print("")
print(">> Train Loss: " + str( np.mean(history.history['loss'])))
print(">> Train ACC: " + str( np.mean(history.history['acc'])))
print(">> Dev ACC: " + str(dev_acc))
print(">> Dev PREC: " + str(dev_precision))
print(">> Dev RECALL: " + str(dev_recall))
print(">> Dev F1_Score: " + str(dev_f1_score))
print(">> Dev CM: " + str(dev_cm))


######### GENERATE OUTPUT PREDICTIONS ####################
dataset_test = pd.read_csv( TEST_CSV)
X_test = df_test.as_matrix()
 
X_test  = StandardScaler().fit_transform( X_test)

print("X_test.shape: {}".format(X_test.shape))

y_hat = model.predict( X_test)

csv = open( OUTPUT_CSV, "w")
csv.write("PassengerId,Survived\n")
for i in range(len(y_hat)):
    csv.write(str(dataset_test.PassengerId[i]) + "," + str( int((y_hat[i] > 0.5))) + "\n")

csv.close()


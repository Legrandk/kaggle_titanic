# -*- coding: utf-8 -*-

# Titanic Challenge: https://www.kaggle.com/c/titanic/data

"""
Este modelo ha logrado mejores resultados:
    Puesto: 5,065 (Top %53)
    Scoring: .77990

La clave ha sido el feature engineering para poder darle mejor informacion a la NN:
    https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

>> TRAIN acc: 0.8385     
>> CM: [[93 17]
        [13 56]]
>> Dev ACC: 0.832402234637
>> Dev PREC: 0.767123287671
>> Dev RECALL: 0.811594202899
>> Dev F1_Score: 0.788732394366
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#DESCOMENTAR ESTO EN KAGGLE!!!!!!!
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#TRAIN_CSV = "../input/train.csv"
#TEST_CSV  = "../input/test.csv"
#OUTPUT_CSV = "survivors_prediction.csv"

TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"
OUTPUT_CSV = "survivors_prediction.csv"


#########################################################

np.random.seed(123)

def create_nn_model( nx = 1, nb_hlayers = 1, nb_neurons = 16, learning_rate = 0.001, weight_decay = 0, weights_init = 'glorot_uniform'):
    np.random.seed(123)

    model = Sequential()
    
    #print(">> N_x: {}, Neurons: {}".format(nx, nb_neurons))
    
    regularizer = None
    if 0 < weight_decay:
        regularizer = regularizers.l2( weight_decay)

    """
    model.add( Dense( units              = nb_neurons,
                      input_dim          = nx,
                      kernel_initializer = weights_init,
                      kernel_regularizer = regularizer,
                      activation         = 'relu'))

    model.add( Dense( units              = nb_neurons,
                      kernel_initializer = weights_init,
                      kernel_regularizer = regularizer,
                      activation         = 'relu'))

    model.add( Dense( units              = nb_neurons,
                      kernel_initializer = weights_init,
                      kernel_regularizer = regularizer,
                      activation         = 'relu'))
    """

    model.add( Dense( units              = nb_neurons,
                      input_dim          = nx,
                      kernel_initializer = weights_init,
                      kernel_regularizer = regularizer,
                      activation         = 'relu'))
    
    model.add( Dense( units              = 1,
                      #input_dim          = nx,
                      kernel_initializer = weights_init,
                      activation         = 'sigmoid'))

    optim = optimizers.Adam(lr = learning_rate) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile( optimizer = optim, #'adam',
                   loss = 'binary_crossentropy',
                   metrics = [ 'accuracy' ])

    return model

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
X_train, X_dev, y_train, y_dev = train_test_split( X, y, test_size=0.2, random_state=0)

# Feature Standardization(mu=0, std=1)
sc_X    = StandardScaler()
X_train = sc_X.fit_transform(X_train)   #(534,853)  => (712,30)
X_dev   = sc_X.transform(X_dev)         #(179,853)  => (179,30)


print("X_train.shape: {}".format(X_train.shape))
print("X_dev.shape: {}".format(X_dev.shape))

# NN MODEL SEARCH

#def create_nn_model( nx = 1, nb_hlayers = 1, nb_neurons = 16, weight_decay = 0, weights_init = 'glorot_uniform'):

model = KerasClassifier(build_fn=create_nn_model, 
                        nx=X_dev.shape[1], 
                        nb_hlayers = 1,
                        nb_neurons = 16)
"""
>> Best: 0.810393 using {'epochs': 400, 'nb_neurons': 18, 'weight_decay': 0, 'nb_hlayers': 1, 'batch_size': 32, 'weights_init': 'glorot_normal'}
>> Best: 0.800562 using {'nb_neurons': 15, 'nb_hlayers': 1, 'batch_size': 32, 'weight_decay': 0, 'weights_init': 'glorot_normal', 'epochs': 400}
>> Best: 0.796348 using {'nb_neurons': 64, 'batch_size': 32, 'weights_init': 'glorot_uniform', 'epochs': 400, 'weight_decay': 0, 'nb_hlayers': 1}
>> Best: 0.792135 using {'weight_decay': 0, 'epochs': 200, 'weights_init': 'glorot_uniform', 'batch_size': 32, 'nb_neurons': 64, 'nb_hlayers': 2}
"""

batch_size = [ 32]
epochs     = [ 275] #, 280, 285, 290, 295, 300]
nb_hlayers = [ 1]
nb_neurons = [ 5] #[2, 3, 4, 5, 6, 7, 8, 9, 10] #np.append([ 18], np.random.random_integers(3,128, size=(1,100)))
learning_rates = pow(10, (-3*np.random.rand(100))-1) #[0.001]
weights_decay = [0.5] #[ 0, 0.5, 0.1, 0.12, 0.15, 0.18, 0.19, 0.20, 0.25, 0.30]
weights_init = ['glorot_normal'] #, 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform', 'random_normal', 'random_uniform']

param_grid = dict( 
        batch_size = batch_size, 
        learning_rate = learning_rates,
        epochs = epochs,
        nb_hlayers = nb_hlayers,
        nb_neurons = nb_neurons,
        weight_decay = weights_decay,
        weights_init = weights_init)

grid = GridSearchCV( estimator = model, param_grid = param_grid, n_jobs = -1)
grid_result = grid.fit( X_train , y_train, verbose = 0)

# summarize results
print(">> Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(">>\t%f (%f) with: %r" % (mean, stdev, param))
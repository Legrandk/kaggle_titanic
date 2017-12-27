# [KAGGLE] Titanic: Machine Learning from Disaster

This NN model prdicts the survivors from Titanic Disaster with 78.9% accuracy.

See more details about competition here https://www.kaggle.com/c/titanic


The best model found was ranked as 0.78947 and got the 3,342 (Top 35%) position.

## Neural Network Architecture: 
 * Input: 29 features
 * Hidden: 9 neurons
 * Hidden: 9 neurons
 * Hidden: 9 neurons
 * Output: 1 neuron

## Hyperparameters
 * Learning rate: 0.001
 * Optimizer: Adam
 * L2 Reg: 0.12
 * Weights Init: Xabier
 * Epochs: 400

    
## Model Performance
 * Dev Confusion Matrix

|      | Dead? | Surv? |
| ---- | ----- | ----- |
| Dead | 103   | 11    |
| Surv | 14    | 51    |
 * Train ACC: 0.830056179775
 * Dev ACC: 0.860335195531
 * Dev PREC: 0.822580645161
 * Dev RECALL: 0.784615384615
 * Dev F1_Score: 0.803149606299
 
        
## Neural Network Models Comparision

 * Model#1 (Simplest):
   * Kaggle Score: 0.76555
   * Architecture: No hidden layers
   * Hyperparameters: Adam(lr=0.001), Epochs=275
   * Best Kagged Score: 0.78947
   * Train Acc: 0.871508379888
   * Dev Acc: 0.828125
   * Dev F1: 0.821705426357
   
 * Model#2 (1 Layer):
   * Kaggle Score: 0.77990
   * Architecture: 1 hidden layer with 5 neurons
   * Hyperparameters: Adam(lr=0.001), Epochs=275
   * Best Kagged Score: 0.78947
   * Train Acc: 0.839724208376
   * Dev Acc: 0.838709677419
   * Dev RECALL: 0.8
   * Dev F1_Score: 0.818897637795
  
 * Model#3 (Deep: 3 Layers):
   * Kaggle Score: 0.78947
   * Architecture: 3 hidden layers with 9 neurons each
   * Hyperparameters: Adam(lr=0.001), L2 Reg=0.12, Epochs=400
   * Best Kagged Score: 0.78947
   * Train Acc: 0.830056179775
   * Dev Acc: 0.860335195531
   * Dev RECALL: 0.784615384615
   * Dev F1_Score: 0.803149606299


## Important Notes
 * nn_gridsearch.py was used to tune the NN model architecture and hyperparameters
 * Feature engineering proposed by Ahmed Besbes https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
   was very useful to improve the information contained in the Name, SibSp, Parch and Cabin features.
   

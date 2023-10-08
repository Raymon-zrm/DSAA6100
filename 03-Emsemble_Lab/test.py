import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_cancer_dataset(num_samples=10, random_seed=4):
    """
    Load a subset of the breast cancer dataset from sklearn, scale it, and return it.
    
    Parameters:
    - num_samples (int): Number of samples to select from the dataset.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - tuple: Scaled feature matrix and target vector.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use random seed for reproducibility
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), num_samples, replace=False)
    return X_scaled[indices], y[indices]

def get_tree_pd(x_train, y_train, model, tree_depth_range):
    """
    Conduct cross-validation for a DecisionTreeClassifier model over different depths.
    
    Parameters:
    - x_train (array-like): Training feature matrix.
    - y_train (array-like): Training target vector.
    - model (DecisionTreeClassifier): Decision Tree model to evaluate.
    - tree_depth_range (range): Range of depths to evaluate the model on.
    
    Returns:
    - DataFrame: Contains the depth and corresponding cross-validation accuracy scores.
    """
    scores_data = {"depth": [], "cv_acc_score": []}
    
    for depth in tree_depth_range:
        model.set_params(max_depth=depth)
        scores = cross_val_score(model, x_train, y_train, cv=5)
        scores_data["depth"].extend([depth] * len(scores))
        scores_data["cv_acc_score"].extend(scores)
    
    return pd.DataFrame(scores_data)


#matplotlib inline

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)

from sklearn.model_selection import learning_curve

cancer_scaled, target = load_cancer_dataset(300, 4)



################################### Train Test split
np.random.seed(40)

#test_proportion
test_prop = 0.2
msk = np.random.uniform(0, 1, len(cancer_scaled)) > test_prop

#Split predictor and response columns
x_train, y_train =  cancer_scaled[msk], target[msk]
x_test , y_test  = cancer_scaled[~msk], target[~msk]

print("Shape of Training Set :", x_train.shape)
print("Shape of Testing Set :" , x_test.shape)

#Your code here
np.random.seed(40)

get_tree_pd(x_train, y_train, DecisionTreeClassifier(), range(1, 31))

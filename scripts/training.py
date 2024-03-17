from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC

from joblib import load
from sklearn.metrics import classification_report, roc_auc_score


class Training: 
    def __init__(self, data):
        self.data = data
    

    def training_process(self):
        df = self.data
        mean_cv_score,X_train, X_test, y_train, y_test = self.cross_validation(df)
        
        best_parameters, best_score, y_pred = self.linear(X_train, y_train, X_test, y_test)
        
        print('The best parameters are: ', best_parameters, ' The best score is ', best_score)
        return(X_train, X_test, y_test, y_pred)
    
    def cross_validation(self, df): 
        X = df.drop('Approved', axis=1)
        y = df['Approved']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        svm_baseline = SVC(random_state=42)
        cv_scores = cross_val_score(svm_baseline, X_train, y_train, cv=5)
        mean_cv_score = cv_scores.mean()

        return(mean_cv_score, X_train, X_test, y_train, y_test)

    def grid_search(self, X_train, y_train): 
        param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.001, 0.01, 0.1, 1]
        }

        grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_

        return(best_parameters, best_score)

    def linear(self, X_train, y_train, X_test, y_test): 
        # Define the parameter grid for LinearSVC
        param_grid = {
            'C': [0.1, 1, 10, 100]
        } 

        # Initialize the LinearSVC
        linear_svc = LinearSVC(dual=False)
        grid_search_linear = GridSearchCV(linear_svc, param_grid, cv=5, scoring='accuracy')

        # Fit it to the data
        grid_search_linear.fit(X_train, y_train) 

        # Best parameters and best score
        best_parameters = grid_search_linear.best_params_
        best_score = grid_search_linear.best_score_
        y_pred_test = grid_search_linear.predict(X_test)
        return(best_parameters, best_score, y_pred_test)


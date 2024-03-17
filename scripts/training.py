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
    
    '''The funcion traing_process use the preprocess data to search results
    for the Linear SVM, also can be done trying with a RBF kernel but in this cas
    the function will return the best parameters and the training results for 
    linear SVM so the only functions that will need is def linear, but the others can also
    be used for compare results.'''

    def training_process(self):
        df = self.data
        mean_cv_score,X_train, X_test, y_train, y_test = self.cross_validation(df)
        #best_parameters, best_score = self.grid_search(X_train, y_train)
        best_parameters, best_score, y_pred = self.linear(X_train, y_train, X_test, y_test)
        #y_test, y_pred = self.svm_training(X_train, y_train, X_test, y_test)
        #y_test, y_pred_sampled = self.applying_smote(X_train, y_train, X_test, y_test)
        
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


    def svm_training(self, X_train, y_train, X_test, y_test):
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        svm_optimized = SVC(C=100, gamma=1, kernel='rbf', random_state=42)
        svm_optimized.fit(X_train_poly, y_train)
        y_pred = svm_optimized.predict(X_test_poly)
        X_test_transformed = poly.transform(X_test)
        y_pred_test = svm_optimized.predict(X_test_transformed)
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test))

        roc_auc = roc_auc_score(y_test, y_pred_test)
        print(f"ROC-AUC Score: {roc_auc}")
        
        return(y_test, y_pred)
    
    def applying_smote(self, X_train, y_train, X_test, y_test):
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        svm_optimized_resampled = SVC(C=100, gamma=1, kernel='rbf', random_state=42)
        svm_optimized_resampled.fit(X_train_resampled, y_train_resampled)
        y_pred_resampled = svm_optimized_resampled.predict(X_test)

        return(y_test, y_pred_resampled)


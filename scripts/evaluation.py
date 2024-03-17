from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

class Evaluation:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred 

    def metrics(self):
        y_test = self.y_test
        y_pred = self.y_pred
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        #parameters=[]
        parameters=[accuracy, precision, recall, f1, roc_auc]
        print(classification_report(y_test, y_pred))
        return(parameters)

    
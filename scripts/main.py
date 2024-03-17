from preprocess import DataPreprocessing
from training import Training
from evaluation import Evaluation

def main():
    
    data_prep = DataPreprocessing('Datasets/clean_dataset.csv')  
    features, preprocess_data = data_prep.preprocess_data()
    model_data = Training(preprocess_data) 
    X_train, X_test, y_test, y_pred = model_data.training_process()

    evaluation_data = Evaluation(y_test, y_pred)  

    p= evaluation_data.metrics()
    print(p)

if __name__ == "__main__":
    main()
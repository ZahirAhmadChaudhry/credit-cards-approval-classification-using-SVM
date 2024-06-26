# Credit Card Approval Classification using SVM

This project is the part of the Machine Learning Algorithms course of our second semester of Machine Learning and Data Mining at Jean Monnet University. This project aims to automate the credit card approval process by employing machine learning algorithms. The Support Vector Machine (SVM) algorithm, known for its effectiveness in classification problems, has been chosen to predict approval statuses based on a provided dataset.

Group Members:

- Zahir AHMAD
- Grisel Quispe Aramayo
- Anna Abrahamyan


## Project Structure

The project is structured as follows:

- `Datasets`: Contains the original (`clean_dataset.csv`) and the preprocessed (`preprocessed_data.csv`) datasets.
- `Diagrams`: This directory holds various diagrams generated during the analysis, including decision surface plots and a radar diagram illustrating model comparisons.
- `scripts`: Contains Python scripts for different stages of the project: `evaluation.py`, `main.py`, `preprocess.py`, and `training.py`.
- `src`: Source directory for Jupyter notebooks and model storage.
- `Models`: Stores the serialized final model (`SVM_Linear.joblib`) and Jupyter notebooks detailing data preprocessing (`Data_Preprocessing.ipynb`) and model training (`Model_Training.ipynb`).
- `LICENSE`: The license file.
- `README.md`: The file you are currently reading.
- `requirements.txt`: A list of Python dependencies required to run the project.

## Setup

To set up the project environment, follow these steps:

1. Ensure you have Python 3.8+ installed on your machine.
2. Clone this repository to your local machine.
3. Navigate to the project directory and install the required dependencies:
   
   ```sh
   pip install -r requirements.txt
   ```

4. To activate the virtual environment:

   On Windows:
   ```sh
   .venv\Scripts\activate
   ```
   
   On Unix or MacOS:
   ```sh
   source .venv/bin/activate
   ```
   
## Usage
The Project can be run from main.py as follows:

Run the `main.py` script to initiate the full pipeline, which includes data preprocessing, model training, and evaluation. This script acts as an entry point for the entire project workflow.

Alternatively, you can explore the Jupyter notebooks in the `src` directory for a more interactive experience.


## License

This project is licensed under the [MIT License](https://github.com/ZahirAhmadChaudhry/credit-cards-approval-classification-using-SVM/blob/main/LICENSE).
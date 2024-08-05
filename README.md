# UCLA - Neural Network Solutions

This project provides a complete pipeline for building, training, evaluating, and optimizing a neural network model to predict the chances of admission based on various factors using Python and popular data science libraries.


## Setup and Installation

1. **Clone the repository:**
   git clone https://github.com/kaurrmanpreett/UCLA-Neural_Network_Solutions.git

2. Create and activate a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required packages:
   pip install -r requirements.txt


## Usage
1. Place your dataset: Ensure your dataset file Admission.csv is placed in the data/ directory.
2. Run the main script: python main.py

## Modules Description
1. imports.py: Contains necessary imports and dependencies.
2. data_preparation.py: Functions to load and preprocess the dataset.
3. model.py: Functions to build, train, and optimize the neural network model.
4. evaluation.py: Functions to evaluate the model and plot the loss curve.

## Results
The script will output the Confusion Matrix and Accuracy for the model on the testing dataset. Additionally, it will plot the loss curve of the model and print the best parameters and score from the grid search.

## License
This project is licensed under the Apache License.

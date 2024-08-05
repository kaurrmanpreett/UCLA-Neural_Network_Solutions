from modules.imports import *
from modules.data_preparation import *
from modules.model import *
from modules.evaluation import *

def main():
    data = load_data('C:\\Users\\kaurr\\OneDrive\\Desktop\\BISI\\2208\\UCLA- Neural Network Solutions\\data\\Admission.csv')
    Xtrain, Xtest, ytrain, ytest = prepare_data(data)
    
    model = build_model()
    trained_model = train_model(model, Xtrain, ytrain)
    
    cm, accuracy = evaluate_model(trained_model, Xtest, ytest)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    
    plot_loss_curve(trained_model)
    
    # Perform Grid Search
    best_params, best_score = grid_search(model, Xtrain, ytrain)
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

if __name__ == '__main__':
    main()

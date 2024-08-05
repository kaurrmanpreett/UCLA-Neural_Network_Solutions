from .imports import confusion_matrix, accuracy_score, plt

def evaluate_model(model, Xtest, ytest):
    ypred = model.predict(Xtest)
    cm = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    return cm, accuracy

def plot_loss_curve(model):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

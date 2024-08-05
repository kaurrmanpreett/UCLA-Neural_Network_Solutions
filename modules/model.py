from .imports import MLPClassifier, GridSearchCV

def build_model(batch_size=50, max_iter=100):
    model = MLPClassifier(batch_size=batch_size, max_iter=max_iter, random_state=123)
    return model

def train_model(model, Xtrain, ytrain):
    model.fit(Xtrain, ytrain)
    return model

def grid_search(model, X, y):
    params = {
        'batch_size': [20, 30, 40, 50],
        'hidden_layer_sizes': [(2,), (3,), (3, 2)],
        'max_iter': [50, 70, 100]
    }
    
    grid = GridSearchCV(model, params, cv=10, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

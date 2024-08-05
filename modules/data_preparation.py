from .imports import pd, train_test_split, MinMaxScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    data = data.drop(['Serial_No'], axis=1)
    data = pd.get_dummies(data, columns=['University_Rating', 'Research'])
    return data

def prepare_data(data):
    X = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    return Xtrain, Xtest, ytrain, ytest

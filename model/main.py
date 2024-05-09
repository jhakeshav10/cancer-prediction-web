import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    # Scale the data to make it uniform
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using LR
    model = LogisticRegression()
    model.fit(X_train,y_train)

    #Test the model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of model: ', accuracy)

    report = classification_report(y_test, y_pred)
    print('Classification Report: \n', report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    

    return model, scaler

def get_clean_data():
    data = pd.read_csv("data/data.csv")

    # print(data.columns)
    # print(data.head())

    data = data.drop(['Unnamed: 32', 'id'], axis = 1)

    data['diagnosis'] = data['diagnosis'].map({ "M": 1, "B": 0 })

    return data

def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()
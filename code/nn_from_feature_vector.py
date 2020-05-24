import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Flatten
from keras.models import model_from_json
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pandas as pd

from sklearn import svm, datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

model_folder = "models/"


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_folder + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_folder + name + ".h5")
    print("Saved model to disk")


def load_model(name):
    # load json and create model
    json_file = open(model_folder + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_folder + name + ".h5")
    print("Loaded model from disk")

    return loaded_model


if __name__ == '__main__':

    features = pd.read_pickle("data/features.pkl")
    features["Sentiment"] = features["Sentiment"].map({1:2, 2:2, 3:3, 4:4, 5:4})

    # Prepare feature vector
    features = features.drop(["Document", "Entity", "Avg_sentiment", "Sd_sentiment", "Doc_length"], axis=1)

    X = (np.array(features.loc[:, features.columns != 'Sentiment']))
    y = (np.array(features['Sentiment']))

    # One hot encode data
    encoder = OneHotEncoder(dtype=int)
    X = encoder.fit_transform(X).toarray()
    y = [[o] for o in y]
    y = encoder.fit_transform(y).toarray()

    print(np.sum(y, axis=0))

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=44)
    print("Train dataset shape: {0}, \nTest dataset shape: {1} \nValidation dataset shape: {2}".format(X_train.shape,
                                                                                                       X_test.shape,
                                                                                                       X_val.shape))

    train = False
    if sys.argv[-1] == "-train":
        train = True

    if train:
        model_ffn = Sequential()
        model_ffn.add(Dense(250, activation='relu', input_dim=X_train.shape[1]))
        model_ffn.add(Dense(3, activation='softmax'))
        model_ffn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        history_ffn = model_ffn.fit(X_train, y_train, epochs=10, batch_size=50, verbose=2)

        save_model(model_ffn, "ffn")

    else:
        model_ffn = load_model("ffn")
        model_ffn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


    # Evaluate model with weighted f1 and keras metrics
    scores_ffn = model_ffn.evaluate(X_test, y_test, verbose=0)

    print("Precision and categorical accuracy: " + str(scores_ffn))

    y_predictions = model_ffn.predict_classes(X_test)
    # Convert One hot encoding back to int classes
    y_test = [np.where(r==1)[0][0] for r in y_test]

    f1_weighted = f1_score(y_test, y_predictions, average='weighted')
    print("F1 weighted score: " + str(f1_weighted))


    # Calculate and show confusion matrix
    cmat = False
    if sys.argv[-1] == "-cmat":
        cmat = True

    if cmat:
        cmat = confusion_matrix(y_test, y_predictions, normalize="true")
        print(cmat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=["negative", "neutral", "positive"])
        disp = disp.plot()

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

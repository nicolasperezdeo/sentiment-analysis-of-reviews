from keras.datasets import imdb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle


def load_imdb_dataset(top_words=5000, max_review_length=500):
    """
    Loads the imdb dataset as provided by Keras (already preprocessed) and saves it to the disk.
    :param top_words: int
    Until which rank of occurrence words are being used
    (e.g. for top_words = 5000 the 5000th most occurring words are used, the rest is omitted.)
    :param max_review_length: int
    Maximum number of words in a review. Review which are shorter than this get zero-padded, longer ones get truncated.
    :return X_train, y_train, X_test, y_test:
    X_train and X_test each contain a list of reviews which themselves are lists of hashed words
    y_train and y_test are lists of labels (positive/negative review)
    """
    try:
        # try to load saved dataset
        X_train = pickle.load(open("./models/NN_x_train.pickle", "rb"))
        y_train = pickle.load(open("./models/NN_y_train.pickle", "rb"))
        X_test = pickle.load(open("./models/NN_x_test.pickle", "rb"))
        y_test = pickle.load(open("./models/NN_y_test.pickle", "rb"))
    except (OSError, IOError):
        # if there are no saved models, load dataset
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

        # truncate and pad input sequences
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        # save the dataset divided into train & test set (and labels)
        pickle.dump(X_train, open("./models/NN_x_train.pickle", "wb"))
        pickle.dump(y_train, open("./models/NN_y_train.pickle", "wb"))
        pickle.dump(X_test, open("./models/NN_x_test.pickle", "wb"))
        pickle.dump(y_test, open("./models/NN_y_test.pickle", "wb"))
    return X_train, y_train, X_test, y_test


def train_model(neural_net,
                X_train,
                y_train,
                X_test,
                y_test,
                top_words=5000,
                max_review_length=500,
                embedding_vecor_length=32,
                number_of_epochs=10,
                batch_size=64,
                silent=True):
    """
    Trains a neural network and saves the model to the disk.
    :param neural_net: String
    The type of neural network to train ('mlp', 'cnn' or 'lstm' are supported right now).
    :param X_train:
    Training dataset.
    :param y_train:
    Labels for the training dataset.
    :param top_words: int
    Omit all words that are not under the top 'top_words' number of words. (That don't occur too often.)
    :param max_review_length: int
    Maximum length of any review.
    :param silent:
    :return:
    """
    # make sure that the neural network name is in lower case
    neural_net = neural_net.lower()

    # there is no need for training if there is already a saved model
    if os.path.exists(f"./models/{neural_net}_{number_of_epochs}epochs.h5") and \
            os.path.exists(f"./models/{neural_net}_{number_of_epochs}epochs.hist"):
        model = load_model(f"./models/{neural_net}_{number_of_epochs}epochs.h5")
        with open(f"./models/{neural_net}_{number_of_epochs}epochs.hist", "rb") as f:
            history = pickle.load(f)
        if not silent:
            print(model.summary())
    else:
        # create the model
        model = Sequential()
        # use a word2vec embedding
        model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

        if neural_net == "mlp":
            # make the input vector one-dimensional
            model.add(Flatten())
            # add one fully connected layer
            model.add(Dense(250, activation='relu'))
        elif neural_net == "cnn":
            # convolutional layer
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            # pooling layer
            model.add(MaxPooling1D(pool_size=2))
            # flatten the vector to be one-dimensional
            model.add(Flatten())
            # a fully connected layer like in the mlp
            model.add(Dense(250, activation='relu'))
        elif neural_net == "lstm":
            # one lstm "layer" (creates internally a whole lstm network)
            model.add(LSTM(100))
        else:
            if not silent:
                print("Neural network of this type is not implemented.")
            # could raise an error here
            return None, None, neural_net

        # finally add an output layer to any neural net
        model.add(Dense(1, activation='sigmoid'))
        # compile the network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not silent:
            print(model.summary())
        # train the model and save the training history
        history = model.fit(
            X_train, y_train,
            epochs=number_of_epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test)
        )

        # save the training history
        with open(f"./models/{neural_net}_{number_of_epochs}epochs.hist", 'wb') as f:
            pickle.dump(history, f)
        # save the trained model for future use
        model.save(f"./models/{neural_net}_{number_of_epochs}epochs.h5")

    return model, history, neural_net


def plot_statistics(history, neural_network, statistic, silent=True):
    """
    Plots the given statistic of a training history.
    :param history: Array
    Saved training history.
    :param neural_network: String
    'cnn', 'lstm' or 'mlp'
    :param statistic: String
    the statistic to plot, e.g. 'loss'
    :param silent: Boolean
    Set to False to let the function print outputs to stdout.
    :return: None
    Just plots.
    """
    if statistic not in ('acc', 'loss'):
        if not silent:
            print(f"{statistic} is not a supported statistic. Try 'acc' or 'loss'.")
        # could raise an error here
        return None

    if statistic == 'acc':
        label = 'Accuracy'
    elif statistic == 'loss':
        label = 'Loss'
    else:
        label = 'unsupported statistic'
        # could raise an error here

    # use a style sheet for the plots
    plt.style.use('ggplot')

    # get statistics for training and test set
    training_stat = history.history[statistic]
    test_stat = history.history[f'val_{statistic}']

    # plot the chosen statistic over the number of trained epochs (for train and test set)
    epochs = range(1, len(training_stat) + 1)
    plt.plot(epochs, training_stat, 'r')
    plt.plot(epochs, test_stat, 'b')

    plt.title(neural_network.upper())

    plt.xlabel('Epoch')
    if statistic == 'loss':
        plt.ylabel('Loss')
        plt.yticks(np.arange(0.0, 0.9, step=0.1))
        plt.legend([f'Train {label}', f'Test {label}'], loc='center right')
    else:
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0.8, 1.025, step=0.025))
        plt.legend([f'Train {label}', f'Test {label}'], loc='lower right')

    plt.xticks(range(1, 11))
    #plt.savefig(f'{neural_network}-{statistic}_10epochs_x.pdf')
    plt.savefig(f'{neural_network}-{statistic}_10epochs_x.png')
    plt.show()


if __name__ == '__main__':
    # fix random seed for reproducibility
    np.random.seed(7)

    # load the imdb movie review dataset as provided by the Keras library
    dataset = load_imdb_dataset()

    # train a multi-layer perceptron on the training set
    _, mlp_hist, _ = train_model('mlp', *dataset)

    # train a CNN on the same dataset and compare the results
    _, cnn_hist, _ = train_model('cnn', *dataset)

    # also train a LSTM
    _, lstm_hist, _ = train_model('lstm', *dataset)

    print('mlp', mlp_hist.history['val_acc'])
    print('cnn', cnn_hist.history['val_acc'])
    print('lstm', lstm_hist.history['val_acc'])

    # plot the accuracies for all networks
    plot_statistics(mlp_hist, 'mlp', 'acc')
    plot_statistics(cnn_hist, 'cnn', 'acc')
    plot_statistics(cnn_hist, 'lstm', 'acc')

    plot_statistics(mlp_hist, 'mlp', 'loss')
    plot_statistics(cnn_hist, 'cnn', 'loss')
    plot_statistics(cnn_hist, 'lstm', 'loss')


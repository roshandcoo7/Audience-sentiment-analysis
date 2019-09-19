{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "O1N7L4SWCb_i",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from tensorflow.keras.preprocessing import sequence\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Embedding\n",
        "from tensorflow.keras.layers import LSTM\n",
        "from tensorflow.keras.datasets import imdb"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kuXZmT14ClEI",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "print('Loading data...')\n",
        "(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nU-DJpfJCyEh",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x_train = sequence.pad_sequences(x_train, maxlen=80)\n",
        "x_test = sequence.pad_sequences(x_test, maxlen=80)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G9XpHWrpCzw8",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model = Sequential()\n",
        "model.add(Embedding(20000, 128))\n",
        "model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))\n",
        "model.add(Dense(1, activation='sigmoid'))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M38DtcIXC4TL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model.compile(loss='binary_crossentropy',\n",
        "              optimizer='adam',\n",
        "              metrics=['accuracy'])"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kRV37NyZC5jW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow.keras.backend import set_session\n",
        "config = tf.ConfigProto()\n",
        "config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU\n",
        "sess = tf.Session(config=config)\n",
        "set_session(sess)  # set this TensorFlow session as the default session for Keras"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GA3Vcg9bC87Z",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model.fit(x_train, y_train,\n",
        "          batch_size=32,\n",
        "          epochs=15,\n",
        "          verbose=2,\n",
        "          validation_data=(x_test, y_test))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "81l0ZUMvW80N",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "score, acc = model.evaluate(x_test, y_test,\n",
        "                            batch_size=32,\n",
        "                            verbose=2)\n",
        "print('Test score:', score)\n",
        "print('Test accuracy:', acc)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
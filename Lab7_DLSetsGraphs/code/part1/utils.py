"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    X_train = np.zeros((n_train, max_train_card), dtype=int)
    y_train = np.zeros(n_train, dtype=int)

    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)

        generated_set = np.random.choice(
            range(1, max_train_card + 1), card, replace=False
        )

        X_train[i, :card] = generated_set
        y_train[i] = np.sum(generated_set)

    return X_train, y_train


def create_test_dataset():

    ############## Task 2
    n_test = 200000
    test_cards = np.arange(5, 101, 5)

    X_test = []
    y_test = []

    for card in test_cards:
        n_samples_per_card = 10000
        X_card = np.zeros((n_samples_per_card, card), dtype=int)
        y_card = np.zeros(n_samples_per_card, dtype=int)
        for i in range(n_samples_per_card):
            generated_set = np.random.choice(range(1, 11), card, replace=True)

            X_card[i] = generated_set
            y_card[i] = np.sum(generated_set)
        X_test.append(X_card)
        y_test.append(y_card)

    return X_test, y_test

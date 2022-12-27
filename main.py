# !/usr/bin/env python
# coding: utf-8

import time
import warnings
import os
import numpy as np
from lambeq import QuantumTrainer, SPSAOptimizer, remove_cups, NumpyModel, Dataset, AtomicType, IQPAnsatz, \
    Sim14Ansatz, Sim15Ansatz, StronglyEntanglingAnsatz, SpiderAnsatz, BobcatParser, PytorchModel, PytorchTrainer
import torch
import sys
import datetime
import json
from utils import read_data_2_labels, read_data_4_labels_quantum, read_data_4_labels_classical
from discopy import Dim

warnings.filterwarnings("ignore")

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def training_classical(n_labels, train_path, val_path, test_path, epochs=40, learning_rate=3e-2, batch_size=30, seed=0):
    train_labels, train_data = '', ''
    dev_labels, dev_data = '', ''
    test_labels, test_data = '', ''

    if n_labels == 2:
        train_labels, train_data = read_data_2_labels(train_path)
        dev_labels, dev_data = read_data_2_labels(val_path)
        test_labels, test_data = read_data_2_labels(test_path)
    elif n_labels == 4:
        train_labels, train_data = read_data_4_labels_classical(train_path)
        dev_labels, dev_data = read_data_4_labels_classical(val_path)
        test_labels, test_data = read_data_4_labels_classical(test_path)

    reader = BobcatParser()

    train_diagrams = reader.sentences2diagrams(train_data)
    dev_diagrams = reader.sentences2diagrams(dev_data)
    test_diagrams = reader.sentences2diagrams(test_data)

    ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(n_labels),
                           AtomicType.SENTENCE: Dim(n_labels)})

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]

    all_circuits = train_circuits + dev_circuits + test_circuits
    model = PytorchModel.from_diagrams(all_circuits)

    sig = torch.sigmoid

    def accuracy(y_hat, y):
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y)) / len(y) / n_labels

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,  # type: ignore
        learning_rate=learning_rate,
        epochs=epochs,
        evaluate_functions={"acc": accuracy},
        evaluate_on_train=True,
        verbose='text',
        seed=seed)

    train_dataset = Dataset(
        train_circuits,
        train_labels,
        batch_size=batch_size)

    dev_dataset = Dataset(dev_circuits, dev_labels)

    trainer.fit(train_dataset, dev_dataset, logging_step=5)

    test_acc = accuracy(model.forward(test_circuits), torch.tensor(test_labels))
    print('Test accuracy:', test_acc.item())


def training_quantum(n_labels, train_path, val_path, test_path, n_layers, ansatz_type, epochs=240, batch_size=10,
                     seed=0):
    train_labels, train_data = '', ''
    dev_labels, dev_data = '', ''
    test_labels, test_data = '', ''
    dim_noun = ''
    dim_sentence = ''

    if n_labels == 2:
        train_labels, train_data = read_data_2_labels(train_path)
        dev_labels, dev_data = read_data_2_labels(val_path)
        test_labels, test_data = read_data_2_labels(test_path)
        dim_noun = 1
        dim_sentence = 1
    elif n_labels == 4:
        train_labels, train_data = read_data_4_labels_quantum(train_path)
        dev_labels, dev_data = read_data_4_labels_quantum(val_path)
        test_labels, test_data = read_data_4_labels_quantum(test_path)
        dim_noun = 2
        dim_sentence = 2

    parser = BobcatParser()

    raw_train_diagrams = parser.sentences2diagrams(train_data)
    raw_dev_diagrams = parser.sentences2diagrams(dev_data)
    raw_test_diagrams = parser.sentences2diagrams(test_data)

    train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
    dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
    test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

    ansatz = ''
    if ansatz_type == 'IQP':
        ansatz = IQPAnsatz({AtomicType.NOUN: dim_noun, AtomicType.SENTENCE: dim_sentence},
                           n_layers=n_layers, n_single_qubit_params=3)
    elif ansatz_type == 'Sim14':
        ansatz = Sim14Ansatz({AtomicType.NOUN: dim_noun, AtomicType.SENTENCE: dim_sentence},
                             n_layers=n_layers, n_single_qubit_params=3)
    elif ansatz_type == 'Sim15':
        ansatz = Sim15Ansatz({AtomicType.NOUN: dim_noun, AtomicType.SENTENCE: dim_sentence},
                             n_layers=n_layers, n_single_qubit_params=3)
    elif ansatz_type == 'Strong':
        ansatz = StronglyEntanglingAnsatz({AtomicType.NOUN: dim_noun, AtomicType.SENTENCE: dim_sentence},
                                          n_layers=n_layers, n_single_qubit_params=3)

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]

    all_circuits = train_circuits + dev_circuits + test_circuits

    model = NumpyModel.from_diagrams(all_circuits, use_jit=True)

    def loss(y_hat, y):
        return -np.sum(y * np.log(y_hat)) / len(y)

    acc = ''
    if n_labels == 2:
        def acc(y_hat, y):
            np.sum(np.round(y_hat) == y) / len(y) / 2
    elif n_labels == 4:
        def acc(y_hat, y):
            y_hat = np.asarray([np.argmax(t) for t in y_hat])
            y = np.asarray([np.argmax(r) for r in y])
            return [i == j for i, j in zip(y_hat, y)].count(True) / len(y)

    trainer = QuantumTrainer(
        model,
        loss_function=loss,
        epochs=epochs,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.2, 'c': 0.06, 'A': 0.1 * epochs},
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        verbose='text',
        seed=seed
    )

    train_dataset = Dataset(
        train_circuits,
        train_labels,
        batch_size=batch_size)

    val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)

    trainer.fit(train_dataset, val_dataset, logging_step=12)

    test_acc = acc(model(test_circuits), test_labels)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':

    old_stdout = sys.stdout
    old_stderr = sys.stderr


    def start_training(settings_path_file):
        settings = ''
        with open(settings_path_file) as json_file:
            settings = json.load(json_file)

        log_file = open("logs/" + settings['name'] + '_' + str(datetime.datetime.now()) + ".log", "w")
        sys.stdout = log_file
        sys.stderr = log_file
        start = time.time()
        print('Start: ' + str(datetime.datetime.now()))
        print(settings)

        # Start Process
        if settings['type'] == 'classical':
            training_classical(settings['labels'], settings['training_path'], settings['validation_path'],
                               settings['testing_path'], settings['epochs'])
        elif settings['type'] == 'quantum':
            training_quantum(settings['labels'], settings['training_path'], settings['validation_path'],
                             settings['testing_path'],
                             settings['number_layers'], settings['ansatz_model'], settings['epochs'])
        # End Process

        end = time.time()
        print("Minutes: " + str(int((end - start) / 60)))
        print('End: ' + str(datetime.datetime.now()))
        log_file.close()

        sys.stderr = old_stderr
        sys.stdout = old_stdout


    # Configure
    settings_path = [ 'settings/quantum/4_labels/sentiment4/2_layers/settings_sentiment_4_labels_3_iqp.json',
        'settings/quantum/4_labels/sentiment5/2_layers/settings_sentiment_5_labels_3_iqp.json']  # TODO

    for x in settings_path:
        start_training(x)

def read_data_2_labels(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1 - t])
            sentences.append(line[1:].strip())
    return labels, sentences


def read_data_4_labels_quantum(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            if t == 0:
                labels.append([[0, 0], [0, 1]])
            elif t == 1:
                labels.append([[0, 0], [1, 0]])
            elif t == 2:
                labels.append([[0, 1], [0, 0]])
            elif t == 3:
                labels.append([[1, 0], [0, 0]])
            sentences.append(line[1:].strip())
    return labels, sentences


def read_data_4_labels_classical(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            if t == 0:
                labels.append([0, 0, 0, 1])
            elif t == 1:
                labels.append([0, 0, 1, 0])
            elif t == 2:
                labels.append([0, 1 ,0, 1])
            elif t == 3:
                labels.append([1, 0, 0, 0])
            sentences.append(line[1:].strip())
    return labels, sentences

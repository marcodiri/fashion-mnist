import numpy as np
from typing import List
from utils.serializer import *


class VotedPerceptron:
    def __init__(self, classes: List, epochs: int):
        self.files = []
        self.classes = classes
        self.epochs = epochs
        self.per_class_percs = []
        self.training_mistakes = 0
        # get files if present
        for c in range(len(classes)):
            filename = 'perceptrons_class' + str(c) + '/' + str(self.epochs) + 'epochs.npy'
            if os.path.isfile(DATA_DIR+filename):
                self.files.append(filename)

    def __train_perceptrons(self, training_list: List, labels_list: np.ndarray, epochs: int):
        """
        Voted perceptron function.

        :param training_list: Array of examples
        :param labels_list: Array of corresponding labels. Each labels_list[i] must be 1 or -1
        :param epochs: Number of epochs
        """

        input_dimension = len(training_list[0])

        k = 0  # error number
        pred_vector = np.zeros(input_dimension, dtype=int)  # hyperplane coefficients (prediction vector)
        weight = 0  # number of correct guesses of the current prediction vector (vote weight)
        perceptrons_list = []

        for epoch in range(epochs):
            for i in range(len(training_list)):
                x = np.array(training_list[i])
                yp = np.sign(np.dot(pred_vector, x))  # prediction
                if yp == labels_list[i]:  # correct prediction
                    weight += 1
                else:  # wrong prediction
                    perceptrons_list.append((pred_vector, weight))  # save previous vector (vector, weight)
                    # update prediction vector summing (if labels_list[i] == 1) or subtracting (if -1) the input vector
                    pred_vector = np.add(pred_vector, labels_list[i]*x)
                    weight = 1  # reset correct guesses number for the new vector (starting from 1 this time?)
                    k += 1
                    self.training_mistakes += 1
            perceptrons_list.append((pred_vector, weight))  # save last vector
            if not (epoch+1) % 10 or epoch == epochs-1:  # serialize every 10 epochs to prevent MemoryError
                yield perceptrons_list
                perceptrons_list.clear()

    def __get_perceptrons_from_file(self, class_num: int):
        if len(self.per_class_percs) > class_num:  # if it was already deserialized just return it
            return self.per_class_percs[class_num]

        filename = 'perceptrons_class' + str(class_num) + '/' + str(self.epochs) + 'epochs.npy'
        if not os.path.isfile(DATA_DIR+filename):
            print("Not yet trained for those parameters")
            raise FileNotFoundError

        perc_list = []
        with Serializer() as serializer:
            try:
                while 1:
                    chunk = serializer.deserialize(filename)
                    # yield vector, weight
                    perc_list += chunk.tolist()
            except FileNotFoundError as e:
                print(e)
            except:  # End Of File reached
                # print("EoF")
                print("Finished deserializing {}".format(filename))
                self.per_class_percs.insert(class_num, perc_list)  # FIXME: this could be a problem with large perc_list
                return perc_list

    def __assign_class_score(self, x: List, method: str):
        score_list = []
        for class_num in range(len(self.classes)):
            perc_list = self.__get_perceptrons_from_file(class_num)
            score = None

            # compute class score based on the chosen method
            if method == 'last':  # use last vector
                score = np.dot(perc_list[-1][0], x)
            elif method == 'vote':  # weighted sum of every vector in the class
                score = sum(c * np.sign(np.dot(v, x)) for v, c in perc_list)
            elif method == 'avg':
                score = sum(c * np.dot(v, x) for v, c in perc_list)

            score_list.insert(class_num, score)

        return score_list

    def predict(self, x: List, method: str):
        score_list = self.__assign_class_score(x, method)
        eval_class = score_list.index(max(score_list))  # get max score index
        perc_list = self.__get_perceptrons_from_file(eval_class)
        weighted_sum = sum(c * np.sign(np.dot(v, x)) for v, c in perc_list)
        prediction = np.sign(weighted_sum)

        return eval_class, prediction

    def train(self, training_list: List, labels_list: List):
        """
        Uses the __train_perceptrons generator function to generate
        the weighted perceptrons for each label in labels_list.

        :param training_list: Array of examples
        :param labels_list: Array of corresponding labels
        """

        # our labels have a 10 element domain {0...9}
        # but since perceptron only accepts binary labels with domain {-1, +1}
        # we gotta reduce each domain element (class) to a binary problem
        # and train perceptrons for each one of these problems.
        # So when we are training on class c we transform each example of
        # the form (Xi, yi) yi∈{0...9} into (Xi, +1) if yi=c
        # or (Xi, -1) if yi≠c

        self.files = []  # empty the files list when retraining
        for curr_class in range(len(self.classes)):
            mkdir('perceptrons_class'+str(curr_class))
            labels_list_normalized = np.array(labels_list, dtype=int)  # copy train array
            for i in range(len(labels_list_normalized)):
                labels_list_normalized[i] = 1 if labels_list_normalized[i] == curr_class else -1

            # yield (vector, weight) as soon as it is created since putting
            # everything into an array requires too much memory
            with Serializer() as serializer:
                filename = 'perceptrons_class' + str(curr_class) + '/' + str(self.epochs) + 'epochs.npy'
                self.files.append(filename)
                for perc_list in self.__train_perceptrons(training_list, labels_list_normalized, self.epochs):
                    serializer.serialize(filename, perc_list)

            print("Finished training for class {}".format(curr_class))


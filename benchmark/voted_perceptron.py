import numpy as np
from typing import List
from utils.serializer import *
from timeit import default_timer as timer


class VotedPerceptron:
    def __init__(self, classes: List, epochs: int):
        self.files = []
        self.classes = classes
        self.epochs = epochs
        self.per_class_percs = []
        # get files if present
        for c in range(len(classes)):
            filename = 'perceptrons_class' + str(c) + '/' + str(self.epochs) + 'epochs.npy'
            if os.path.isfile(MODEL_SAVE_DIR+filename):
                self.files.append(filename)

    def __train_perceptrons(self, X: List, Y: np.ndarray, T: int):
        """
        Voted perceptron function.

        This is a generator that yields the tuple (vector, weight) as soon as the current
        prediction vector makes an error or the examples are finished. This means that only one prediction vector
        is kept in memory at any time. Actually keeping a list of all prediction vectors would run into MemoryError.

        :param X: Array of examples
        :param Y: Array of corresponding labels. Each Y[i] must be 1 or -1
        :param T: Number of epochs
        """

        input_dimension = len(X[0])

        k = 0  # numero di errori
        V = np.zeros(input_dimension, dtype=int)  # coefficienti iperpiano (vettore di previsione)
        c = 0  # numero di guesses corrette del vettore V attuale (peso votazione)

        for t in range(T):
            for i in range(len(X)):
                x = np.array(X[i])
                yp = np.sign(np.dot(V, x))  # previsione
                if yp == Y[i]:  # previsione corretta
                    c += 1
                else:  # previsione errata
                    yield V, c  # ritorna vettore precedente (vettore, peso)
                    V = np.add(V, Y[i]*x)  # aggiorna vettore di previsione sommandogli/sottraendogli il vettore di input
                    c = 1  # resetta numero di guesses corrette per nuovo vettore (stavolta parte da 1?)
                    k += 1
            yield V, c  # ritorna ultimo vettore

    def __get_perceptrons_from_file(self, class_num: int):
        if len(self.per_class_percs) > class_num:  # if it was already deserialized just return it
            return self.per_class_percs[class_num]

        filename = 'perceptrons_class' + str(class_num) + '/' + str(self.epochs) + 'epochs.npy'
        if not os.path.isfile(MODEL_SAVE_DIR+filename):
            print("Not yet trained for those parameters")
            raise FileNotFoundError

        perc_list = []
        with Serializer() as serializer:
            try:
                while 1:
                    vector, weight = serializer.deserialize(filename)
                    # yield vector, weight
                    perc_list.append((vector, weight))
            except FileNotFoundError as e:
                print(e)
            except:  # End Of File reached
                # print("EoF")
                print("Finished deserializing {}".format(filename))
                self.per_class_percs.insert(class_num, perc_list)  # FIXME: this could be a problem with large perc_list
                return perc_list

    def __assign_class_score(self, x: List, method: str):
        best_class = None
        best_score = None
        best_perc_list = None
        for class_num in range(len(self.classes)):
            perc_list = self.__get_perceptrons_from_file(class_num)

            if method == 'last':
                score = np.dot(perc_list[-1][0], x)  # use last vector to compute score
                if best_score is None or score > best_score:
                    best_class = class_num
                    best_score = score
                    best_perc_list = perc_list

        return best_class, best_perc_list

    def predict(self, x: List, method: str):
        eval_class, perc_list = self.__assign_class_score(x, method)
        weighted_sum = sum(c * np.sign(np.dot(v, x)) for v, c in perc_list)

        return eval_class, np.sign(weighted_sum)

    def train(self, X: List, Y: List):
        """
        Uses the voted_perceptron generator function to generate the weighted perceptrons for each label in Y.

        Every tuple (vector, weight) yielded by the generator is yielded along with the current class back to the caller.

        :param X: Array of examples
        :param Y: Array of corresponding labels
        :param T: Number of epochs
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
            Y_new = np.array(Y, dtype=int)  # copy train array
            for i in range(len(Y_new)):
                Y_new[i] = 1 if Y_new[i] == curr_class else -1

            # yield (vector, weight) as soon as it is created since putting
            # everything into an array requires too much memory
            with Serializer() as serializer:
                filename = 'perceptrons_class' + str(curr_class) + '/' + str(self.epochs) + 'epochs.npy'
                self.files.append(filename)
                for vector, weight in self.__train_perceptrons(X, Y_new, self.epochs):
                    serializer.serialize(filename, (vector, weight))



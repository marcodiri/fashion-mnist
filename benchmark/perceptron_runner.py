from configs import DATA_DIR
from utils import mnist_reader
from timeit import default_timer as timer
from benchmark.voted_perceptron import VotedPerceptron

X_train, Y_train = mnist_reader.load_mnist(path=DATA_DIR, kind='train')
X_test, Y_test = mnist_reader.load_mnist(path=DATA_DIR, kind='t10k')

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

N_EPOCHS = 10

vp = VotedPerceptron(CLASS_NAMES, N_EPOCHS)


def train():
    start = timer()

    vp.train(X_train, Y_train)

    end = timer()
    print(end - start)


def predict():
    start = timer()

    error_count = 0
    correct_count = 0
    for i in range(100):
        start2 = timer()
        eval_class, prediction = vp.predict(X_test[i], 'last')
        if prediction and eval_class == Y_test[i]:
            correct_count += 1
        else:
            error_count += 1
        print("{}: Evaluating class {} to {} and it was {}".format(i, eval_class, prediction == 1, Y_test[i]))
        end2 = timer()
        print("Exec time: {}".format(end2 - start2))

    end = timer()
    print("Correct: {}, Incorrect: {}".format(correct_count, error_count))
    print("Exec time: {}".format(end - start))


train()
# predict()


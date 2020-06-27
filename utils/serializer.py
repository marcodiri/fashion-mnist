import os
import numpy as np
from configs import MODEL_SAVE_DIR, DATA_DIR, touch_dir

DATA_DIR = MODEL_SAVE_DIR + DATA_DIR.split('/')[-1] + "/"


def mkdir(base_dir: str = None):
    touch_dir(DATA_DIR)
    if base_dir is not None:
        touch_dir(DATA_DIR+base_dir)


class Serializer:
    def __init__(self):
        self.f = None
        self.filename = None
        mkdir()  # create save folder if not present

    def __enter__(self):
        return self

    #  closes file when Serializer is deleted
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if self.f is not None and not self.f.closed:
            self.f.close()

    def __set_file(self, filename: str):
        if self.f is None:
            self.filename = filename
            self.f = open(DATA_DIR+filename, "wb")
        elif self.filename != filename:
            self.filename = filename
            self.__exit__()
            self.f = open(DATA_DIR+filename, "wb")

    def __get_file(self, filename: str):
        if self.f is None:
            self.filename = filename
            self.f = open(DATA_DIR+filename, "rb")
        elif self.filename != filename:
            self.filename = filename
            self.__exit__()
            self.f = open(DATA_DIR+filename, "rb")

    def serialize(self, filename, data):
        """
        Serialize data into file.

        :param filename: the file to output the serialized data to
        :param data: the data to be serialized
        """
        self.__set_file(filename)
        np.save(self.f, data, allow_pickle=True)

    def deserialize(self, filename: str):
        """
        Deserialize saved items into arrays.

        :param filename: the name of the file to be deserialized
        """
        if os.path.isfile(DATA_DIR + filename):
            self.__get_file(filename)
            return np.load(self.f, allow_pickle=True)
        else:
            raise FileNotFoundError("File not found")


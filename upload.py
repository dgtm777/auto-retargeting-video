from constants import root_dir
import pickle
import os


def my_upload():
    """
    Функция, загружающая нейронную сеть
    """
    filename_path = os.path.join(root_dir, "model/best_model_cpu.pkl")
    a_file = open(
        filename_path,
        "rb",
    )
    my_files = pickle.load(a_file)
    return my_files

import tensorflow as tf
import pandas as pd
import numpy as np

from predicode.datasets.imagedata import ImageData

class Cifar10(ImageData):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self):
        cifar_data = tf.keras.datasets.cifar10.load_data()[0]
        labels = pd.DataFrame({
            'label': [int(lab[0]) for lab in cifar_data[1]],
            'label_text': np.array([Cifar10.labels[int(lab[0])] for lab in cifar_data[1]])
        })
        super().__init__(cifar_data[0], labels)
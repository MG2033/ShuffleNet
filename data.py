from keras.datasets import cifar100
import numpy as np


class DataLoader:
    """Data Loader class. As a simple case, the model is tried on CIFAR100. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

    def load_data(self):
        # No validation set was taken for CIFAR100. Feel free to make the appropriate changes for your datasets.
        (_, self.y_train), (self.X_test, self.y_test) = cifar100.load_data()

        # For going in the same experiment as the paper. Resizing the input image data to 224x224 is done.
        self.X_train = np.zeros((50000, 224, 224, 3), dtype=np.uint8)

        self.train_data_len = self.X_train.shape[0]
        self.test_data_len = self.X_test.shape[0]
        img_height = self.X_train.shape[1]
        img_width = self.X_train.shape[2]
        num_channels = self.X_train.shape[3]
        return img_height, img_width, num_channels, self.train_data_len, self.test_data_len

    def generate_batch(self, train=True):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if train:
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size], 0]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        else:
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size, 0]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.train_data_len:
                    start_idx = 0
                yield X_batch, y_batch

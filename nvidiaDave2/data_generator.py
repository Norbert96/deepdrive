import glob
import h5py
import tensorflow as tf
import cv2
import numpy as np
# hdf5_path = '/media/norbi/DATA/autonomus_database/deepdrive/linux_recordings/2018-01-18__05-14-48PM'


class generator:
    def __init__(self, files):
        self.files = files

    def __call__(self):
        for file in self.files:
            with h5py.File(file, 'r') as hf:
                # print(hf.keys())
                for frame in hf.keys():
                    img = hf[frame]['camera_00000']['image']
                    img = convert_to_normal_picture(img)
                    img = convert_to_nvidia_size(img)

                    steering = dict(hf[frame].attrs)['steering']
                    yield (img, steering)


def convert_to_nvidia_size(img, width=200, height=66):
    orig_height = img.shape[0]
    img = img[orig_height - (height * 2): orig_height, :]
    img = cv2.resize(img, (width, height))

    return img


def convert_to_normal_picture(img):
    img = np.array(img, dtype='float32')
    img += np.array([104., 117., 123.], np.float32)
    img = np.array(img, dtype='uint8')
    return img


def get_hdf5_file_names(path):
    files = glob.glob(path + '/**/*.hdf5', recursive=True)
    return files

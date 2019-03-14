
import glob
import h5py
import tensorflow as tf
import cv2
import numpy as np
hdf5_path = '/media/norbi/DATA/autonomus_database/deepdrive/linux_recordings/2018-01-18__05-14-48PM'


class generator:
    def __init__(self, files):
        self.files = files

    def __call__(self):
        for file in self.files:
            with h5py.File(file, 'r') as hf:
                print(hf.keys())
                for frame in hf.keys():
                    img = hf[frame]['camera_00000']['image']
                    img = convert_to_normal_picture(img)
                    # img = convert_to_nvidia_size(img)

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
    # if train:
    files = files[1:]
    # else:
    #     files = files[0:1]
    # if len(files) == 0:
    #     raise Exception('zero %s hdf5 files, aborting!' % 'train' if train else 'eval')
    return files


def get_all_frame(path):
    file = h5py.File(path, 'r')
    images = []
    for i in file:
        images.append(path + ' ' + i)
    return images


def get_frame(frame_name):

    path = frame_name.spit()[0]
    name = frame_name.spit()[1]
    file = h5py.File(path, 'r')
    frame = file[name]
    return frame


def load_and_preprocess_image(frame_name):
    frame_name = frame_name.decode()
    print(frame_name)
    frame = get_frame(frame_name)
    return frame['cameras'][0]['image']


import time
g = generator(get_hdf5_file_names(hdf5_path))
a = g()
for i in a:
    time.sleep(0.5)
    cv2.imshow('frame', i[0])
    print(i[1])
    cv2.waitKey(1)

# image_tuples = get_all_frame(get_hdf5_file_names(hdf5_path)[0])
# print(image_tuples)
# path_ds = tf.data.Dataset.from_tensor_slices(image_tuples)

# image_ds = path_ds.map(load_and_preprocess_image)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 8))
# for n, image in enumerate(image_ds.take(4)):
#     plt.subplot(2, 2, n + 1)
#     plt.imshow(image)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(caption_image(all_image_paths[n]))

# print(path_ds)

# print(get_hdf5_file_names(hdf5_path))
# print(get_all_frame(get_hdf5_file_names(hdf5_path)[0]))

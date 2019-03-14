
import glob
import h5py
import tensorflow as tf
hdf5_path = '/media/norbi/DATA/autonomus_database/deepdrive/linux_recordings/2018-01-18__05-14-48PM'


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
    frame = get_frame(frame_name)
    return frame['cameras'][0]['image']


image_tuples = get_all_frame(get_hdf5_file_names(hdf5_path)[0])
print(image_tuples)
path_ds = tf.data.Dataset.from_tensor_slices(image_tuples)

image_ds = path_ds.map(load_and_preprocess_image)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n + 1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))

print(path_ds)

# print(get_hdf5_file_names(hdf5_path))
# print(get_all_frame(get_hdf5_file_names(hdf5_path)[0]))

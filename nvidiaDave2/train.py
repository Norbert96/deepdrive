
import data_generator as dg
import model as m
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras

hdf5_path = '/media/norbi/DATA/autonomus_database/deepdrive/linux_recordings/2018-01-18__05-14-48PM'


# def map_fn(image, label):
#       '''Preprocess raw data to trainable input. '''
#     x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
#     y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
#     return x, y

def run():
    files = dg.get_hdf5_file_names(hdf5_path)
    training_files = files[:-1]
    evaluate_files = []
    evaluate_files.append(files[-1])
    ds = tf.data.Dataset.from_generator(
        dg.generator(training_files), (tf.int8, tf.float32), (tf.TensorShape([66, 200, 3]), tf.TensorShape([])))

    # dataset = dataset.map(map_fn)
    ds = ds.batch(46)
    ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    model = m.create_model()
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer="adam", loss="mse")
    iterator = ds.make_one_shot_iterator()

    # print(iterator.get_next())

    tbCallback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                             write_graph=True, write_images=True)

    tbCallback.set_model(model)
    cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/cp.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)

    # validation
    print(evaluate_files)
    dt = tf.data.Dataset.from_generator(
        dg.generator(training_files), (tf.int8, tf.float32), (tf.TensorShape([66, 200, 3]), tf.TensorShape([])))
    dt = dt.batch(1)
    dt = dt.repeat()

    dt = dt.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    model.fit(iterator, steps_per_epoch=1000, validation_data=dt.make_one_shot_iterator(), validation_steps=1000, epochs=15, verbose=1, callbacks=[tbCallback, cp_callback])
    # model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    #                     samples_per_epoch=NB_SAMPLES, nb_epoch=NB_EPOCH,
    #                     validation_data=(X_val, y_val))


if __name__ == "__main__":
    # files = dg.get_hdf5_file_names(hdf5_path)
    # print(dg.generator(files).calculate_all_picture())
    run()

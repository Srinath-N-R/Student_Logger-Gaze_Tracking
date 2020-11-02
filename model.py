import tensorflow as tf
from tensorflow import keras

train_dir_eye = 'train\eye'
train_datagen_eye = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_eye = train_datagen_eye.flow_from_directory(train_dir_eye, target_size = (27,180), batch_size = 50, class_mode = 'binary')

train_dir_face = 'train\ce'
train_datagen_face = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_face = train_datagen_face.flow_from_directory(train_dir_face, target_size = (480,640), batch_size = 50, class_mode = 'binary')

validation_dir_eye = 'validation\eye'
validation_datagen_eye = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_data_eye = validation_datagen_eye.flow_from_directory(validation_dir_eye, target_size = (27,180), batch_size = 5, class_mode = 'binary')

validation_dir_face = 'validation\ce'
validation_datagen_face = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_data_face = validation_datagen_face.flow_from_directory(validation_dir_face, target_size = (480,640), batch_size = 5, class_mode = 'binary')

model1 = keras.Sequential([keras.layers.Conv2D(16, (3,3), input_shape = (27,180,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(32, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(64, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Flatten(), keras.layers.Dense(512, activation='relu')])
# first = tf.keras.Sequential()
# second = tf.keras.Sequential()
# first = keras.layers.Dense(1, input_shape=((512,)))(model1)
# first = keras.layers.LeakyReLU(alpha=0.1)(first)
# first = keras.layers.Dropout(0.5)(first)
model2 = keras.Sequential([keras.layers.Conv2D(16, (3,3), input_shape = (480,640,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(32, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(64, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Flatten(), keras.layers.Dense(512, activation='relu')])
# second = keras.layers.Dense(1, input_shape=((512,)))(model2)
# second = keras.layers.LeakyReLU(alpha=0.1)(second)
# second = keras.layers.Dropout(0.5)(second)
avg = keras.layers.average([model1.output, model2.output])
aver = keras.layers.Dense(256, activation="relu")(avg)
# conv = keras.layers.concatenate(([first, second]))
conv = keras.layers.Flatten()(aver)
dense = keras.layers.Dense(512)(conv)
dens = keras.layers.LeakyReLU(alpha=0.1)(dense)
den = keras.layers.Dropout(0.5)(dens)
output = keras.layers.Dense(1, activation = 'sigmoid')(den)
# eye_input = keras.layers.Input(shape=(150,150,3))
# face_input = keras.layers.Input(shape=(150,150,3))
model = keras.models.Model(inputs=[model1.input, model2.output], outputs=[output])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01), loss = 'binary_crossentropy', metrics = ['acc'])

best_weights_file="weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]

model.fit_generator([train_data_eye, train_data_face], steps_per_epoch=50, verbose=1, epochs=50, validation_data=[validation_data_eye, validation_data_face], validation_steps=11)


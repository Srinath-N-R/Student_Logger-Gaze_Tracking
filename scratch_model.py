import tensorflow as tf
from tensorflow import keras

train_dir_eye = 'train\eye'
# train_datagen_eye = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_data_eye = train_datagen_eye.flow_from_directory(train_dir_eye, target_size = (27,180), batch_size = 50, class_mode = 'binary')

train_dir_face = 'train\ce'
# train_datagen_face = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_data_face = train_datagen_face.flow_from_directory(train_dir_face, target_size = (480,640), batch_size = 50, class_mode = 'binary')

validation_dir_eye = 'validation\eye'
# validation_datagen_eye = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# validation_data_eye = validation_datagen_eye.flow_from_directory(validation_dir_eye, target_size = (27,180), batch_size = 5, class_mode = 'binary')

validation_dir_face = 'validation\ce'
# validation_datagen_face = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# validation_data_face = validation_datagen_face.flow_from_directory(validation_dir_face, target_size = (480,640), batch_size = 5, class_mode = 'binary')
model1 = keras.Sequential([keras.layers.Conv2D(16, (3,3), input_shape = (27,180,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(32, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(64, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Flatten(), keras.layers.Dense(512, activation='relu')])
model2 = keras.Sequential([keras.layers.Conv2D(16, (3,3), input_shape = (480,640,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(32, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Conv2D(64, (3,3), activation='relu'), keras.layers.MaxPool2D(2,2), keras.layers.Flatten(), keras.layers.Dense(512, activation='relu')])
avg = keras.layers.average([model1.output, model2.output])
aver = keras.layers.Dense(256, activation="relu")(avg)
conv = keras.layers.Flatten()(aver)
dense = keras.layers.Dense(128)(conv)
dens = keras.layers.LeakyReLU(alpha=0.1)(dense)
den = keras.layers.Dropout(0.5)(dens)
output = keras.layers.Dense(1, activation = 'sigmoid')(den)
model = keras.models.Model(inputs=[model1.input,model2.input], outputs=[output])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01), loss = 'binary_crossentropy', metrics = ['acc'])

best_weights_file="weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]


input_imgen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,)

def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height1, img_width1, img_height2, img_width2):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height1, img_width1),
                                          class_mode='categorical',
                                          batch_size=batch_size)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height2, img_width2),
                                          class_mode='categorical',
                                          batch_size=batch_size)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


inputgenerator = generate_generator_multiple(generator=input_imgen,
                                             dir1=train_dir_eye,
                                             dir2=train_dir_face,
                                             batch_size=50,
                                             img_height1=27,
                                             img_width1=180,
                                             img_height2=480,
                                             img_width2=640
                                             )

testgenerator = generate_generator_multiple(input_imgen,
                                            dir1=validation_dir_eye,
                                            dir2=validation_dir_face,
                                            batch_size=5,
                                            img_height1=27,
                                            img_width1=180,
                                            img_height2=480,
                                            img_width2=640)

history = model.fit_generator(inputgenerator,
                              steps_per_epoch= 50,
                              epochs=50,
                              validation_data=testgenerator,
                              validation_steps= 11,
                              use_multiprocessing=True,
                              shuffle=False, callbacks=callbacks)


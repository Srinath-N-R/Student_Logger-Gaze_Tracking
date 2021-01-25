import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

df = pd.read_csv('data.txt', header=None, sep=' ')
X = df.iloc[:, :21]
y = df.iloc[:, 21]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=666)

input = Input(21)
layer = Dense(21, activation='relu')(input)
layer = Dense(20, activation='relu')(layer)
layer = Dense(10, activation='relu')(layer)
layer = Dense(5, activation='relu')(layer)
output = Dense(1, activation='sigmoid')(layer)

model = Model(inputs = input, outputs = output)

checkpoint_name = 'Weights-{epoch:03d}--{val_accuracy:.5f}--{accuracy:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='accuracy', save_best_only = True, mode ='auto')
callbacks = [checkpoint]

model.compile(optimizer= Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size = 50,
                    epochs=10000,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test),
                    shuffle = True
                    )

model.save('Model.h5')
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adadelta,Adamax,Adagrad,RMSprop,Nadam,TFOptimizer,Adam
from keras.callbacks import TensorBoard,CSVLogger
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model

batch_size = 128
epochs = 1000
learning_rate = 0.01

def model1():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_adam.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_adam.json','w').write(json_string)
    model.save_weights('detect_adam.h5')

    with open("history_detect_adam.pickle",mode='wb') as f:
        pickle.dump(history.history,f)


    plot_model(model, to_file="model_detect_adam.png", show_shapes=True)

    return model


def model2():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_SGD.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_SGD.json','w').write(json_string)
    model.save_weights('detect_SGD.h5')

    with open("history_detect_SGD.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_SGD.png", show_shapes=True)

    return model

def model3():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_adadelta.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_adadelta.json','w').write(json_string)
    model.save_weights('detect_adadelta.h5')

    with open("history_detect_adadelta.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_adadelta.png", show_shapes=True)

    return model

def model4():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adagrad',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_adagrad.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_adagrad.json','w').write(json_string)
    model.save_weights('detect_adagrad.h5')

    with open("history_detect_adagrad.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_adagrad.png", show_shapes=True)

    return model

def model5():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_RMSprop.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_RMSprop.json','w').write(json_string)
    model.save_weights('detect_RMSprop.h5')

    with open("history_detect_RMSprop.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_RMSprop.png", show_shapes=True)

    return model

def model6():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Nadam',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_nadam.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_Nadam.json','w').write(json_string)
    model.save_weights('detect_Nadam.h5')

    with open("history_detect_Nadam.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_Nadam.png", show_shapes=True)

    return model

def model7():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='TFOptimizer',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_TFOptimizer.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_TFOptimizer.json','w').write(json_string)
    model.save_weights('detect_TFOptimizer.h5')

    with open("history_detect_TFOptimizer.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_TFOptimizer.png", show_shapes=True)

    return model

def model8():
    model = Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])

    model.summary()

    tbcb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True)
    csv_logger = CSVLogger('training_log_adamax.csv')

    train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory('./train',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('./test',target_size=(128,128),batch_size=batch_size,class_mode='categorical')
    print(train_generator.class_indices)


    history1 = model.fit_generator(train_generator,
                                   samples_per_epoch=1000,
                                   nb_epoch=epochs,
                                   validation_data=validation_generator,
                                   callbacks = [tbcb,csv_logger],
                                   nb_val_samples=1000)

    

    json_string = model.to_json()
    open('detect_adamax.json','w').write(json_string)
    model.save_weights('detect_adamax.h5')

    with open("history_detect_adamax.pickle",mode='wb') as f:
        pickle.dump(history1.history,f)


    plot_model(model, to_file="model_detect_adamax.png", show_shapes=True)

    return model
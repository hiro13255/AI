from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD,Adamax
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
from keras.utils import plot_model
from keras.callbacks import TensorBoard,CSVLogger
import pickle
from model import model1,model2,model3,model4,model5,model6,model7,model8


model1()
model2()
model3()
model4()
model5()
model6()
model8()
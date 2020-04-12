#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
#%% batches

train_path='C:\\Users\\apmys\\Desktop\\vgg16_2\\Training'
validation_path='C:\\Users\\apmys\\Desktop\\vgg16_2\\Testing'
testing_path='C:\\Users\\apmys\\Desktop\\vgg16_2\\test_images'

train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['dosa','idli','mysore bonda','mysore pak','paratha','poha','rasam'],batch_size=20)
valid_batches=ImageDataGenerator().flow_from_directory(validation_path,target_size=(224,224),classes=['dosa','idli','mysore bonda','mysore pak','paratha','poha','rasam'],batch_size=16)
test_batches= ImageDataGenerator().flow_from_directory(testing_path,target_size=(224,224),classes=['dosa','idli','mysore bonda','mysore pak','paratha','poha','rasam'],batch_size=7)
                                                      
model = Sequential()   
model.add(Conv2D(32, (3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
##compile
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%%Confusion matrix

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(cm)
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0],cm.shape[1])):
        plt.txt(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

#%% training    

model.fit_generator(train_batches,
steps_per_epoch = 2382/20,
epochs = 10,
validation_data = valid_batches,
validation_steps = 982/16)

model.summary()
#%%

from __future__ import print_function
import tensorflow.keras.utils
from tensorflow.keras import utils as np_utils
import tensorflow.keras.applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential

vgg16_model=tensorflow.keras.applications.vgg16.VGG16()
type(vgg16_model)  #to check type of model of vgg16 it is of type model

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable=False
    
model.add(Dense(7,activation='softmax'))

model.summary()

#%%train the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,
steps_per_epoch = 2382/20,
epochs = 10,
validation_data = valid_batches,
validation_steps = 982/16)

model.save('finetune4.h5')
model.summary()

#%%Confusion matrix


test_imgs,test_labels=next(test_batches)
prediction=model.predict_generator(test_batches,steps=1,verbose=0)

cm = confusion_matrix(test_labels.argmax(axis=1), prediction.argmax(axis=1))

cm_plot_labels=['dosa','idli','mysore bonda','mysore pak','paratha','poha','rasam']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')

print(prediction)
np.set_printoptions(precision=2)
#%%
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import model_from_yaml
from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_vgg16


newmodel=load_model('finetune4.h5')
image = load_img('test_images/idli/idli1.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
print(image.shape)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print(image.shape)
    
image = preprocess_input(image)
print(image.shape)
yhat = newmodel.predict(image)

pred=np.sort(yhat)
print(pred)
print(pred[0][1])
print(yhat)
np.set_printoptions(suppress=True)

if yhat[0][0]>=0.5:
    print('dosa')
    print(yhat[0][0]*100,'%')
    np.set_printoptions(suppress=True)
    
elif  yhat[0][1]>=0.5:
    print('idli')
    print(yhat[0][1]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][2]>=0.5:
    print('mysore bonda')
    print(yhat[0][2]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][3]>=0.5:
    print('mysore pak')
    print(yhat[0][3]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][4]>=0.5:
    print('paratha')
    print(yhat[0][4]*100,'%')
    np.set_printoptions(suppress=True)
    
elif yhat[0][5]>=0.5:
    print('poha')
    print(yhat[0][5]*100,'%')
    np.set_printoptions(suppress=True)
    
else :
    print('rasam')
    print(yhat[0][6]*100,'%')
    np.set_printoptions(suppress=True)
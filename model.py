#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


#%%
tf.debugging.set_log_device_placement(True)

# # %%
# 
# for dirname, _, filenames in os.walk('/dataset'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# %%
from tensorflow import keras
import seaborn as sns
import pandas as pd
import cv2

# %%
test_csv = pd.read_csv('E:/ML/DL DanceClassification/dataset/test.csv')
train_csv = pd.read_csv('E:/ML/DL DanceClassification/dataset/train.csv')

# %%
train_csv['target'].value_counts().plot(kind = 'bar')

# %%
base = 'E:/ML/DL DanceClassification/dataset'
train_dir = os.path.join(str(base)+'/train/')
test_dir = os.path.join(str(base)+'/test/')

# %%
train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

# %%
img_width = 224
img_height = 224

# %%
def training_data_prep(list_name_images, train_csv, train_dir):
    train_data = []
    train_label = []
    for image_name in list_name_images:
        train_data.append(cv2.resize(cv2.imread(train_dir+image_name), 
        (img_width, img_height), interpolation = cv2.INTER_CUBIC))
        if image_name in list(train_csv['Image']):
            train_label.append(train_csv.loc[train_csv['Image'] == image_name, 'target'].values[0])
    return train_data, train_label


# %%
def test_data_prep(list_name_images, train_dir):
    test_data = []
    for image_name in list_name_images:
        test_data.append(cv2.resize(cv2.imread(test_dir+image_name), (img_width, img_height),
        interpolation = cv2.INTER_CUBIC))
    return test_data

# %%
training_data, training_labels = training_data_prep(train_fnames, train_csv, train_dir)

# %%
training_data[:5]

# %%
training_labels[:5]

# %%
def show_img_batch(image_batch, label_batch):
    plt.figure(figsize=(12, 12))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n].title())
        plt.axis('off')

# %%
show_img_batch(training_data, training_labels)

# %%
testing_data = test_data_prep(test_fnames, test_dir)

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
training_labels = encoder.fit_transform(training_labels)

# %%
training_labels[:10]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data, 
                                    training_labels, 
                                    test_size = 0.3, random_state = 42)

# # %%
# #Data Augmentation
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagenerator=ImageDataGenerator(rescale=1./255,
#                                       rotation_range=40,
#                                       zoom_range=0.20,
#                                       width_shift_range=0.10,
#                                        height_shift_range=0.10,
#                                        horizontal_flip=True,)

# test_datagenerator=ImageDataGenerator(rescale=1./255)


# train_datagenerator.fit(X_train)
# test_datagenerator.fit(X_test)
# test_datagenerator.fit(testing_data)

# X_train=np.array(X_train)
# testing_data=np.array(testing_data)
# X_test=np.array(X_test)

# # %%
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# # %%
# from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

# #%%
# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('accuracy')>=0.98):
#       print("\nReached 98% accuracy so cancelling training!")
#       self.model.stop_training = True

# #%%
# callbacks = myCallback()

# # %%
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape = (224, 224,3)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(218, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(8, activation = 'softmax')
# ])


# # %%
# from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.callbacks import ReduceLROnPlateau
# from tensorflow.python.keras.applications.vgg16 import VGG16
# model.compile(
#             optimizer='adam', 
#             loss='categorical_crossentropy', 
#             metrics=['accuracy'])

#     # model fitting
# history = model.fit(
#     train_datagenerator.flow(X_train, to_categorical(y_train,8)),
#     epochs=15,
#     validation_data=test_datagenerator.flow(X_test, to_categorical(y_test,8)),
#     batch_size=16,
#     steps_per_epoch=10,
#     verbose=1)


# # %%
# plt.plot(history.history['accuracy'],label='accuracy')
# plt.legend(loc='best')
# plt.show()

# %%
#########################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
########################################################
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagenerator = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        rotation_range=40,  
        zoom_range = 0.20,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=True,  
        vertical_flip=False) 


val_datagenerator=ImageDataGenerator(
        rescale=1. / 255
)

train_datagenerator.fit(X_train)
val_datagenerator.fit(X_test)
X_train=np.array(X_train)
X_test=np.array(X_test)

#%%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%%
from tensorflow.keras.applications.vgg16 import VGG16
vggmodel =VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3),pooling='max')

 # Print the model summary
vggmodel.summary()

#%%
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau

vggmodel.trainable = False
model = Sequential([
  vggmodel, 
  Dense(1024, activation='relu'),
  Dropout(0.2),
  Dense(512, activation='relu'),
  Dropout(0.2),
  Dense(256, activation='relu'),
  Dropout(0.2),
  Dense(8, activation='softmax'),
])
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
#%%
with tf.device('/GPU:0'):
    from tensorflow.keras.utils import to_categorical
    model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    history =model.fit_generator(
        train_datagenerator.flow(X_train, to_categorical(y_train,8), batch_size=16),
        steps_per_epoch= 254// 16, 
        validation_data=val_datagenerator.flow(X_test, to_categorical(y_test,8), batch_size=16),
        validation_steps=110 // 16,
        verbose=1,
        epochs=50,
        callbacks=[callbacks]
    )
#%%
model.save_weights('model_saved.h5')

# %%
history.history['val_accuracy']

# %%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

acc      = history.history['accuracy']
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[ 'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot( epochs, acc )
plt.plot( epochs, val_acc )
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')

# %%
val_datagenerator.fit(testing_data)
testing_data = np.array(testing_data)

# %%
predictions = model.predict(testing_data)

# %%
predictions

# %%
predictions=[np.argmax(i) for i in predictions]
predictions

# %%
target=encoder.inverse_transform(predictions)
target

# %%
submission = pd.DataFrame({ 'Image': test_csv.Image, 'target': target })
submission.to_csv('output2.csv', index=False)
submission

# %%

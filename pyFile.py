import os
import random
import numpy as np
import tensorflow as tf
import matplotlib as plt

# Image Processing Libraries
from tqdm import tqdm  # to show processing bar
from skimage.io import imread, imshow
from skimage.transform import resize

''' Standard Size of Images '''
img_width = 128
img_hight = 128
img_channels = 3
# adjust randome seed size
seed = 42
np.random.seed = seed

'''
Data Preprocessing
'''
train_path = 'stage1_train/'
test_path = 'stage1_test/'
train_ids = next(os.walk(train_path))[1]#next return a tuple(all folders at 1 position
test_ids = next(os.walk(test_path))[1]

X_train = np.zeros((len(train_ids),img_hight, img_width, img_channels), dtype=np.uint8)
# X_train.shape
Y_train = np.zeros((len(train_ids),img_hight, img_width, 1), dtype=np.bool) # for Masks
# Y_train.shape
#
""" Resizing Training Images  """
#
print('Resizing Training Images and Masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = train_path + id_
    image = imread(path + '/images/'+ id_+'.png')[:,:,:img_channels]
    image = resize(image, (img_hight, img_width), mode='constant', preserve_range= True)
    X_train[n] = image # fill empty X_train with values from images
    mask = np.zeros((img_hight, img_width, 1), dtype=np.bool)
    for mask_file in next(os.walk(path+'/masks/'))[2]:
        test_mask = imread(path + '/masks/' + mask_file)
        test_mask = np.expand_dims(resize(test_mask,(img_hight, img_width), mode='constant', preserve_range=True), axis=-1)#
        mask = np.maximum(mask, test_mask)
    Y_train[n] = mask


X_test = np.zeros((len(test_ids),img_hight, img_width, img_channels), dtype=np.uint8)
# X_test.shape
sizes_test = []
print('Resizing Test Images...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = test_path + id_ + '/images/' + id_ + '.png'
    image = imread(path)[:,:,:img_channels]
    sizes_test.append([image.shape[0], image.shape[1]])
    image = resize(image, (img_hight, img_width), mode='constant', preserve_range=True)
    X_test[n] = image
print('Resizing Done!!!')

# Ploting Some Random Images
# image_x = 658
image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x]) # showing image
plt.show()
imshow(np.squeeze(Y_train[image_x])) # show coresponding mask
plt.show()


# Build the Model
inputs = tf.keras.layers.Input((img_width, img_hight, img_channels)) # Input Layer
float_Values = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # converting each value into flot for Neural Network
# Contraction path (going down)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(float_Values)
c1 = tf.keras.layers.Dropout(0.1)(c1) # dropout of 10%
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) # creating new Layer
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)  # MaxPolling layer

c2 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout((0.2))(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)

#Expensive Path (Up Scalling)
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)##
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)##
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (2,2), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (2,2), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
# Output layer
outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(c9)

# Compiling Model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')#, matrices=['accuracy'])  #optimize contais  Back propagation

model.summary()

# ModelCheckpoints
checkPointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkPointer
]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)#

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
print('Actual Spot Nuclei Image:')
imshow(X_train[ix])
plt.show()
print('Expected Infected Region:')
imshow(np.squeeze(Y_train[ix]))
plt.show()
print('Predicted Result:')
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# perform a sanity check on some random validating samples
ix = random.randint(0,len(preds_val_t))
print('Actual Spot Nuclei Image:')
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
print('Expected Infected Region:')
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
print('Predicted Result:')
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

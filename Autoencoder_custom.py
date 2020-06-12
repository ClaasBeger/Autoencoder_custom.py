import os
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISBLE_DEVICES"] = "1" #train model on gpu 1

from scipy.io import loadmat
mnist = loadmat(r"C:\Users\user\Spydercnnect\mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

mnist_data1,mnist_data2,mnist_data3,mnist_data4,mnist_data5,mnist_data6,mnist_data7,mnist_data8,mnist_data9,mnist_data10 = np.split(mnist_data,10)
test_data = np.concatenate([np.split(mnist_data1,4)[3],np.split(mnist_data2,4)[3],np.split(mnist_data3,4)[3],np.split(mnist_data4,4)[3],np.split(mnist_data5,4)[3],np.split(mnist_data6,4)[3],np.split(mnist_data7,4)[3],np.split(mnist_data8,4)[3],np.split(mnist_data9,4)[3],np.split(mnist_data10,4)[3]])
train_data =  np.concatenate([np.split(mnist_data1,4)[0:3],np.split(mnist_data2,4)[0:3],np.split(mnist_data3,4)[0:3],np.split(mnist_data4,4)[0:3],np.split(mnist_data5,4)[0:3],np.split(mnist_data6,4)[0:3],np.split(mnist_data7,4)[0:3],np.split(mnist_data8,4)[0:3],np.split(mnist_data9,4)[0:3],np.split(mnist_data10,4)[0:3]])
train_data = train_data[0, :, :]
mnist_labels1,mnist_labels2,mnist_labels3,mnist_labels4,mnist_labels5,mnist_labels6,mnist_labels7,mnist_labels8,mnist_labels9,mnist_labels10 = np.split(mnist_label,10)
test_labels = np.concatenate([np.split(mnist_labels1,4)[3],np.split(mnist_labels2,4)[3],np.split(mnist_labels3,4)[3],np.split(mnist_labels4,4)[3],np.split(mnist_labels5,4)[3],np.split(mnist_labels6,4)[3],np.split(mnist_labels7,4)[3],np.split(mnist_labels8,4)[3],np.split(mnist_labels9,4)[3],np.split(mnist_labels10,4)[3]])
train_labels =  np.concatenate([np.split(mnist_labels1,4)[0:3],np.split(mnist_labels2,4)[0:3],np.split(mnist_labels3,4)[0:3],np.split(mnist_labels4,4)[0:3],np.split(mnist_labels5,4)[0:3],np.split(mnist_labels6,4)[0:3],np.split(mnist_labels7,4)[0:3],np.split(mnist_labels8,4)[0:3],np.split(mnist_labels9,4)[0:3],np.split(mnist_labels10,4)[0:3]])
print("Training set (images) shape: {shape}".format(shape=train_data.shape))
#Training set (images) shape: (60000, 28,28)
# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_data.shape))
label_dict = {
  0: 'Zero',
  1: 'One',
  2: 'Two',
  3: 'Three',
  4: 'Four',
  5: 'Five',
  6: 'Six',
  7: 'Seven',
  8: 'Eight',
  9: 'Nine',
}

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(test_data[0], (28,28))
curr_lbl = test_labels[0]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# ((17500, 28, 28, 1), (1750, 28, 28, 1))
train_data = train_data.reshape(-1, 28,28, 1)
test_data = test_data.reshape(-1, 28,28, 1)

print(train_data.shape)
print(test_data.shape)

#verify the training and testing data types
print(train_data.dtype)
print(test_data.dtype)
#(dtype('uint8'), dtype('uint8'))

#rescale the training and testing data with the maximum pixel value
print(np.max(train_data), np.max(test_data))
#(255.0, 255.0)
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

#verify training and testing data (1.0 after rescaling)
print(np.max(train_data), np.max(test_data))
#(1.0, 1.0)


plt.subplot(122)
curr_img = np.reshape(test_data[1699], (28,28))
curr_lbl = test_labels[1699]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")



# Training data already aligns (val_loss doesnt change because of other reasons ?)
# occurrences = np.count_nonzero(train_labels == 0)
# print(test_data.size/test_data[0].size)
# print(test_labels.size)
# print("Occurrences of 0", occurrences)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data, 
                                                             test_size=0.2, 
                                                             random_state=13)

batch_size = 128  #vmutable
epochs = 20
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    """
    

    Parameters
    ----------
    input_img : 28 x 28 x 1 matrix (wide and thin)
        One of the input images.

    Returns
    -------
    decoded : 2D convoluted Layer
        The created Layer.

    """
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded
# After model is created,compile it using the optimizer to be RMSProp.

#specify the loss type as the mean squared error,loss after every batch will be computed between batch of predicted output and ground truth using mean squared error pixel by pixel:

autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()

# #train model with keras fit
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

# #Training vs Validation plot
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

pred = autoencoder.predict(test_data)
print(pred.shape)


#search one specimen for each number in order
inputNumbers = np.zeros(10,int)
for g in range(10):
    #search for suitable val
 #   ind = np.where(train_labels == g)[0][0]
    inputNumbers[g] = np.where(test_labels == g)[0][0]
print(inputNumbers)


# #TODO: adjust the test and predicted image with one of each type or
# # at least multiple (maybe first()-func?)
# #show original Test images vs predicted images
# plt.figure(figsize=(20, 4))
# print("Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(test_data[i, ..., 0], cmap='gray')
#     curr_lbl = test_labels[i]
#     plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# plt.show()    
# plt.figure(figsize=(20, 4))
# print("Reconstruction of Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(pred[i, ..., 0], cmap='gray')  
# plt.show()
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_data[inputNumbers[i],...,0], cmap='gray')
    curr_lbl = test_labels[inputNumbers[i]]
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[inputNumbers[i], ...,0], cmap='gray')  
plt.show()
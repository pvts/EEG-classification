#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Group 24--- Machine Learning Course, Tilburg University, Block II/2019-2020
#ML Speech Classification Challenge

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, MaxPooling1D, Dropout
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, recall_score, precision_score


# In[ ]:


#Import the data
b = np.load('feat.npy', allow_pickle = True)
path = np.load('path.npy', allow_pickle = True)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:


#Import MFCC files and merge with train data
df_npy = pd.DataFrame()
df_npy['Mel Freq'] = b
df_npy['Audiofile'] = path

train = train.rename(columns={"path": "Audiofile", "word": "Target"})
df_train = df_npy.merge(train, on="Audiofile", how = 'inner')
mel = df_train["Mel Freq"]
mel_np = np.array(mel)


# In[ ]:


#Empty array for padding the features
newarray = np.zeros([94824,99,13])


# In[ ]:


for index, y in enumerate(mel_np):
    for i, u in enumerate(y):
        for j, o in enumerate(u):
            newarray[index][i][j] = o
            
mel_freq = newarray


# In[ ]:


#Target labels
targets = []
target = df_train["Target"]
for t in target:
    targets.append(t)


# In[ ]:


#Splitting the dataset into a train and test set
X = mel_freq
y = targets 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


# One-hot Indicator array for classes
onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)

label_encoder = LabelEncoder()
integer_encoded_y_val = label_encoder.fit_transform(y_val)


# In[ ]:


#Transform the labels
y_val = to_categorical(integer_encoded_y_val)


# In[ ]:


#Using Early Stopping to reduce overfitting
checkpoint = ModelCheckpoint(filepath = 'bestmodel.hdf5', mode='auto', save_best_only=True,
                             restore_best_weights=True, verbose=1)
Early_Stopping = EarlyStopping(monitor='val_accuracy', patience=30)


# In[ ]:


#Defining callbacks and our validation set
callbacks = [Early_Stopping, checkpoint]
validation= [X_val, y_val]


# In[ ]:


#Model
input_shape = X_train[0].shape #99, 13

model = Sequential()
model.add(Conv1D(filters = 128, kernel_size = 8, activation='relu', input_shape = input_shape))
model.add(MaxPooling1D(13))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(435, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(335))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(235))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(135))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(35, activation='softmax'))

#optimizer = Adam(lr=0.003)

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics = ['accuracy'])
final_model = model.fit(X_train, Y_train, epochs=150, batch_size=60, verbose=1, 
                        validation_data=validation,callbacks=callbacks)


# In[ ]:


model.load_weights("bestmodel.hdf5")


# In[ ]:


#Calcuting accuracy and loss
score = model.evaluate(X_val, y_val, verbose=0)

print('Validation loss:',score[0])
print('Validation accuracy:', score[1])


# In[ ]:


y_pred = model.predict_classes(X_val,verbose = False)
y_pred = to_categorical(y_pred)


# In[ ]:


#Creating 1D arrays to plot
y_val_plt = y_val.argmax(axis=1)
y_pred_plt = y_pred.argmax(axis=1)

#Visualize the expected vs the predicted output
output = pd.DataFrame()
output['Expected Output'] = y_val_plt
output['Predicted Output'] = y_pred_plt
#output.head(10)


# In[ ]:


#Plotting train vs validation accuracy
history = final_model 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[ ]:


#Plotting train vs validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.plot()


# In[ ]:


#Calculating precision
print(precision_score(y_val, y_pred, average = 'macro'))

#Calculating recall
print(recall_score(y_val, y_pred, average = 'macro'))


# In[ ]:


#Merge the test set with MFCC files
test = test.rename(columns={"path": "Audiofile"})
df_test = test.merge(df_npy, on="Audiofile", how = 'inner')
mel_test = df_test["Mel Freq"]
mel_np_test = np.array(mel_test)


# In[ ]:


#Zero array for padding for the test set
newarray = np.zeros([11005,99,13])

for index, y in enumerate(mel_np_test):
    for i, u in enumerate(y):
        for j, o in enumerate(u):
            newarray[index][i][j] = o
            
test_array = newarray 


# In[ ]:


#Use the model the make predictions on the test set
y_test = model.predict_classes(test_array)
results = label_encoder.inverse_transform(y_test)

final = pd.DataFrame()
final["path"] = df_test["Audiofile"]
final["word"] = results


# In[ ]:


#Create the final results.csv file
final.to_csv('result.csv', index=False)


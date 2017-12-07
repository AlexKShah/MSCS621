# Alex Shah
# Language Classifier
# Adapted under CC - Lucas KM

## Tested with:
## Tensorflow 1.4.0
## Keras 2.0.8
## Python 3.5.2
## Checkpoint error resolved with:
## sudo apt install libhdf5-dev
## sudo pip3 install h5py

import os
import re
import math
import random
import collections
import time
import numpy as np
import matplotlib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.optimizers
from keras.utils import plot_model

## VARIABLES ##
#languages
languages_dict = {'en':0,'fr':1}
#sample length
text_sample_size = 70
# number of samples per language
num_lang_samples = 1000000

def decode_langid(langid):    
    for dname, did in languages_dict.items():
        if did == langid:
            return dname

def size_mb(size):
    size_mb =  '{:.2f}'.format(size/(1000*1000.0))
    return size_mb + " MB"
    
## Special Characters ##
def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?¿¡'
    french = 'àâæçéèêêîïôœùûüÿ'
    all_lang_chars = base_en + french
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort() 
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    letters_string = ''
    letters = small_chars + big_chars
    for letter in letters:
        letters_string += letter
    return small_chars,big_chars,letters_string

alphabet = define_alphabet()

## Directories ##
data_directory = "./data/"
source_directory = data_directory + 'source'
cleaned_directory = data_directory + 'cleaned'
samples_directory = data_directory + 'samples'
train_test_directory = data_directory + 'train_test'
chkpts_directory = data_directory + 'chkpts'

for filename in os.listdir(source_directory):
    path = os.path.join(source_directory, filename)
    if not filename.startswith('.'):
        print("Using:")
        print((path), "size : ",size_mb(os.path.getsize(path)))

def remove_xml(text):
    return re.sub(r'<[^<]+?>', '', text)

def remove_newlines(text):
    return text.replace('\n', ' ') 
    
def remove_manyspaces(text):
    return re.sub(r'\s+', ' ', text)

def clean_text(text):
    text = remove_xml(text)
    text = remove_newlines(text)
    text = remove_manyspaces(text)
    return text
    
for lang_code in languages_dict:
    path_src = os.path.join(source_directory, lang_code+".txt")
    f = open(path_src)
    content = f.read()
    f.close()
    # cleaning
    content = clean_text(content)
    path_cl = os.path.join(cleaned_directory,lang_code + '_cleaned.txt')
    f = open(path_cl,'w')
    f.write(content)
    f.close()
    del content
print ("END OF CLEANING")

def get_sample_text(file_content,start_index,sample_size):
    while not (file_content[start_index].isspace()):
        start_index += 1
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index+sample_size 
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]

def count_chars(text,alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts

def get_input_row(content,start_index,sample_size):
    sample_text = get_sample_text(content,start_index,sample_size)
    counted_chars_all = count_chars(sample_text.lower(),alphabet[0])
    counted_chars_big = count_chars(sample_text,alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts
    
path = os.path.join(cleaned_directory, lang_code+"_cleaned.txt")
with open(path, 'r') as f:
    content = f.read()
    random_index = random.randrange(0,len(content)-2*text_sample_size)
    sample_text = get_sample_text(content,random_index,text_sample_size)
    sample_input_row = get_input_row(content,random_index,text_sample_size)
    input_size = len(sample_input_row)
    del content

sample_data = np.empty((num_lang_samples*len(languages_dict),input_size+1),dtype = np.uint16)
lang_seq = 0
jump_reduce = 0.2 
for lang_code in languages_dict:
    start_index = 0
    path = os.path.join(cleaned_directory, lang_code+"_cleaned.txt")
    with open(path, 'r') as f:
        file_content = f.read()
        content_length = len(file_content)
        remaining = content_length - text_sample_size*num_lang_samples
        jump = int(((remaining/num_lang_samples)*3)/4)
        for idx in range(num_lang_samples):
            input_row = get_input_row(file_content,start_index,text_sample_size)
            sample_data[num_lang_samples*lang_seq+idx,] = input_row + [languages_dict[lang_code]]
            start_index += text_sample_size + jump
        del file_content
    lang_seq += 1

np.random.shuffle(sample_data)

path_smpl = os.path.join(samples_directory,"lang_samples_"+str(input_size)+".npz")
np.savez_compressed(path_smpl,data=sample_data)
del sample_data

path_smpl = os.path.join(samples_directory,"lang_samples_"+str(input_size)+".npz")
dt = np.load(path_smpl)['data']
random_index = random.randrange(0,dt.shape[0])
bins = np.bincount(dt[:,input_size])
#for lang_code in languages_dict: 
#    print (lang_code,bins[languages_dict[lang_code]])

dt = dt.astype(np.float64)
X = dt[:,0:input_size]
Y = dt[:,input_size]
del dt
random_index = random.randrange(0,X.shape[0])

#takes a while to generate
time.sleep(10)

standard_scaler = preprocessing.StandardScaler().fit(X)
X = standard_scaler.transform(X)   

Y = keras.utils.to_categorical(Y, num_classes=len(languages_dict))

seed = 42
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
del X, Y

#takes a while to clear
time.sleep(10)
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(input_size)+".npz")
np.savez_compressed(path_tt,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
del X_train,Y_train,X_test,Y_test

path_tt = os.path.join(train_test_directory,"train_test_data_"+str(input_size)+".npz")
train_test_data = np.load(path_tt)
X_train = train_test_data['X_train']
Y_train = train_test_data['Y_train']
X_test = train_test_data['X_test']
Y_test = train_test_data['Y_test']
del train_test_data

print("TRAINING")

model = Sequential()
model.add(Dense(500,input_dim=input_size,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(300,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(100,kernel_initializer="glorot_uniform",activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(len(languages_dict),kernel_initializer="glorot_uniform",activation="softmax"))
model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=model_optimizer,
              metrics=['accuracy'])
filepath="chkpts/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
callbacks_list = [checkpoint,tensorboard]
history = model.fit(X_train,Y_train,
          epochs=5,
          validation_split=0.10,
          batch_size=64,
          callbacks=callbacks_list,
          verbose=2,
          shuffle=True)

scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

Y_pred = model.predict_classes(X_test)
Y_pred = keras.utils.to_categorical(Y_pred, num_classes=len(languages_dict))

target_names =  list(languages_dict.keys())
print(classification_report(Y_test, Y_pred, target_names=target_names))

en_text = "You are welcome, most noble Sorceress, to the land of the Munchkins. We are so grateful to you \
for having killed the Wicked Witch of the East, and for setting our people free from bondage."
fr_text = "Voilà cinq mois que j'en faisais fonction, et, ma foi, je supportais bien cette responsabilité et \
goûtais fort cette indépendance. Je puis même affirmer, sans me flatter"

text_texts_array = [en_text,fr_text]
test_array = []
for item in text_texts_array:
    cleaned_text = clean_text(item)
    input_row = get_input_row(cleaned_text,0,text_sample_size)
    test_array.append(input_row)

test_array = standard_scaler.transform(test_array)     
Y_pred = model.predict_classes(test_array)
for id in range(len(test_array)):
    print ("Text:",text_texts_array[id][:50],"... -> Predicted lang: ", decode_langid(Y_pred[id]))

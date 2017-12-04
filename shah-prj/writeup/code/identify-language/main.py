# imports
import os
import re
import math
import random
import collections
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.optimizers
from keras.utils import plot_model

# key variables
# dictionary of languages that our classifier will cover
languages_dict = {'en':0,'es':1}
# length of cleaned text used for training and prediction - 140 chars
text_sample_size = 140
# number of language samples per language that we will extract from source files
num_lang_samples = 50000

# utility function to turn language id into language code
def decode_langid(langid):    
    for dname, did in languages_dict.items():
        if did == langid:
            return dname

# utility function to return file Bytes size in MB
def size_mb(size):
    size_mb =  '{:.2f}'.format(size/(1000*1000.0))
    return size_mb + " MB"

# we will use alphabet for text cleaning and letter counting
def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?¿¡'
    spanish = 'áéíóúüñ'
    all_lang_chars = base_en + spanish
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

# I keep raw data in 'original' subfolder, and cleaned data in 'cleaned' subfolder
# 'samples' subdirectory is for files with text samples processed according to my sampling procedure
# 'train_test' subdirecrory is for files with np.arrays prepared for NN train and test data (both features and targets)
data_directory = "./data/"
source_directory = data_directory + 'source'
cleaned_directory = data_directory + 'cleaned'
samples_directory = data_directory + 'samples'
train_test_directory = data_directory + 'train_test'

for filename in os.listdir(source_directory):
    path = os.path.join(source_directory, filename)
    if not filename.startswith('.'):
        print((path), "size : ",size_mb(os.path.getsize(path)))
        
# we will create here several text-cleaning procedures. 
# These procedure will help us to clean the data we have for training, 
# but also will be useful in cleaning the text we want to classify, before the classification by trained DNN

# remove XML tags procedure
# for example, Wikipedia Extractor creates tags like this below, we need to remove them
# <doc id="12" url="https://en.wikipedia.org/wiki?curid=12" title="Anarchism"> ... </doc>
def remove_xml(text):
    return re.sub(r'<[^<]+?>', '', text)

# remove new lines - we need dense data
def remove_newlines(text):
    return text.replace('\n', ' ') 
    
# replace many spaces in text with one space - too many spaces is unnecesary
# we want to keep single spaces between words
# as this can tell DNN about average length of the word and this may be useful feature
def remove_manyspaces(text):
    return re.sub(r'\s+', ' ', text)

# and here the whole procedure together
def clean_text(text):
    text = remove_xml(text)
    text = remove_newlines(text)
    text = remove_manyspaces(text)
    return text
    
for lang_code in languages_dict:
    path_src = os.path.join(source_directory, lang_code+".txt")
    f = open(path_src)
    content = f.read()
    print('Language : ',lang_code)
    print ('Content before cleaning :-> ',content[1000:1000+text_sample_size])
    f.close()
    # cleaning
    content = clean_text(content)
    print ('Content after cleaning :-> ',content[1000:1000+text_sample_size])
    path_cl = os.path.join(cleaned_directory,lang_code + '_cleaned.txt')
    f = open(path_cl,'w')
    f.write(content)
    f.close()
    del content
    print ("Cleaning completed for : " + path_src,'->',path_cl)
    print (100*'-')
print ("END OF CLEANING")


# this function will get sample of texh from each cleaned language file. 
# It will try to preserve complete words - if word is to be sliced, sample will be shortened to full word
def get_sample_text(file_content,start_index,sample_size):
    # we want to start from full first word
    # if the firts character is not space, move to next ones
    while not (file_content[start_index].isspace()):
        start_index += 1
    #now we look for first non-space character - beginning of any word
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index+sample_size 
    # we also want full words at the end
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]

# we need only alpha characters and some (very limited) special characters
# exactly the ones defined in the alphabet
# no numbers, most of special characters also bring no value for our classification task
# (like dot or comma - they are the same in all of our languages so does not bring additional informational value)

# count number of chars in text based on given alphabet
def count_chars(text,alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts

# process text and return sample input row for DNN
# note that we are counting separatey:
# a) counts of all letters regardless of their size (whole text turned to lowercase letter)
# b) counts of big letters only
# this is because German uses big letters for beginning of nouns so this feature is meaningful
def get_input_row(content,start_index,sample_size):
    sample_text = get_sample_text(content,start_index,sample_size)
    counted_chars_all = count_chars(sample_text.lower(),alphabet[0])
    counted_chars_big = count_chars(sample_text,alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts
    
# let's see if our processing is returning counts
# last part calculates also input_size for DNN so this code must be run before DNN is trained
path = os.path.join(cleaned_directory, "es_cleaned.txt")
with open(path, 'r') as f:
    content = f.read()
    random_index = random.randrange(0,len(content)-2*text_sample_size)
    sample_text = get_sample_text(content,random_index,text_sample_size)
    print ("1. Sample text: \n",sample_text)
    print ("2. Reference alphabet: \n",alphabet[0],alphabet[1])
    sample_input_row = get_input_row(content,random_index,text_sample_size)
    print ("3. Sample_input_row: \n",sample_input_row)
    input_size = len(sample_input_row)
    print ("4. Input size : ", input_size)
    del content

# now we have preprocessing utility functions ready. Let's use them to process each cleaned language file
# and turn text data into numerical data samples for our neural network
# prepare numpy array
sample_data = np.empty((num_lang_samples*len(languages_dict),input_size+1),dtype = np.uint16)
lang_seq = 0
jump_reduce = 0.2 # part of characters removed from jump to avoid passing the end of file
for lang_code in languages_dict:
    start_index = 0
    path = os.path.join(cleaned_directory, lang_code+"_cleaned.txt")
    with open(path, 'r') as f:
        print ("Processing file : " + path)
        file_content = f.read()
        content_length = len(file_content)
        remaining = content_length - text_sample_size*num_lang_samples
        jump = int(((remaining/num_lang_samples)*3)/4)
        print ("File size : ",size_mb(content_length),\
               " | # possible samples : ",int(content_length/input_size),\
              "| # skip chars : " + str(jump))
        for idx in range(num_lang_samples):
            input_row = get_input_row(file_content,start_index,text_sample_size)
            sample_data[num_lang_samples*lang_seq+idx,] = input_row + [languages_dict[lang_code]]
            start_index += text_sample_size + jump
        del file_content
    lang_seq += 1
    print (100*"-")
     
# let's randomy shuffle the data
np.random.shuffle(sample_data)
# reference input size
print ("Input size : ",input_size )
print (100*"-")
print ("Samples array size : ",sample_data.shape )
path_smpl = os.path.join(samples_directory,"lang_samples_"+str(input_size)+".npz")
np.savez_compressed(path_smpl,data=sample_data)
print(path_smpl, "size : ",size_mb(os.path.getsize(path_smpl)))
del sample_data

# now we will review the data  - control check step
path_smpl = os.path.join(samples_directory,"lang_samples_"+str(input_size)+".npz")
dt = np.load(path_smpl)['data']
random_index = random.randrange(0,dt.shape[0])
print ("Sample record : \n",dt[random_index,])
print ("Sample language : ",decode_langid(dt[random_index,][input_size]))
# we can also check if the data have equal share of different languages
print ("Dataset shape :", dt.shape)
bins = np.bincount(dt[:,input_size])
print ("Language bins count : ") 
for lang_code in languages_dict: 
    print (lang_code,bins[languages_dict[lang_code]])

# we need to preprocess data for DNN yet again - scale it 
# scling will ensure that our optimization algorithm (variation of gradient descent) will converge well
# we need also ensure one-hot econding of target classes for softmax output layer
# let's convert datatype before processing to float
dt = dt.astype(np.float64)
# X and Y split
X = dt[:,0:input_size]
Y = dt[:,input_size]
del dt
# random index to check random sample
random_index = random.randrange(0,X.shape[0])
print("Example data before processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])
time.sleep(120) # sleep time to allow release memory. This step is very memory consuming
# X preprocessing
# standar scaler will be useful laterm during DNN prediction
standard_scaler = preprocessing.StandardScaler().fit(X)
X = standard_scaler.transform(X)   
print ("X preprocessed shape :", X.shape)
# Y one-hot encoding
Y = keras.utils.to_categorical(Y, num_classes=len(languages_dict))
# See the sample data
print("Example data after processing:")
print("X : \n", X[random_index,])
print("Y : \n", Y[random_index])
# train/test split. Static seed to have comparable results for different runs
seed = 42
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
del X, Y
# wait for memory release again
time.sleep(120)
# save train/test arrays to file
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(input_size)+".npz")
np.savez_compressed(path_tt,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
print(path_tt, "size : ",size_mb(os.path.getsize(path_tt)))
del X_train,Y_train,X_test,Y_test

# load train data first from file
path_tt = os.path.join(train_test_directory,"train_test_data_"+str(input_size)+".npz")
train_test_data = np.load(path_tt)
X_train = train_test_data['X_train']
print ("X_train: ",X_train.shape)
Y_train = train_test_data['Y_train']
print ("Y_train: ",Y_train.shape)
X_test = train_test_data['X_test']
print ("X_test: ",X_test.shape)
Y_test = train_test_data['Y_test']
print ("Y_test: ",Y_test.shape)
del train_test_data

# create DNN using Keras Sequential API
# I added Dropout to prevent overfitting
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

# let's fit the data
# history variable will help us to plot results later
history = model.fit(X_train,Y_train,
          epochs=12,
          validation_split=0.10,
          batch_size=64,
          verbose=2,
          shuffle=True)

# now we will face the TRUTH. What is our model real accuracy tested on unseen data?
scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# and now we will prepare data for scikit-learn classification report
Y_pred = model.predict_classes(X_test)
Y_pred = keras.utils.to_categorical(Y_pred, num_classes=len(languages_dict))

# and run the report
target_names =  list(languages_dict.keys())
print(classification_report(Y_test, Y_pred, target_names=target_names))

# show plot accuracy changes during training
plt.plot(history.history['acc'],'g')
plt.plot(history.history['val_acc'],'r')
plt.title('accuracy across epochs')
plt.ylabel('accuracy level')
plt.xlabel('# epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# show plot of loss changes during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Frank Baum, The Wonderful Wizard of Oz, Project Gutenberg, public domain
en_text = "You are welcome, most noble Sorceress, to the land of the Munchkins. We are so grateful to you \
for having killed the Wicked Witch of the East, and for setting our people free from bondage."
# Fernando Callejo Ferrer, Música y Músicos Portorriqueños, Project Gutenberg, public domain
es_text = "Dedicada esta sección a la reseña de los compositores nativos y obras que han producido, con ligeros \
comentarios propios a cada uno, parécenos oportuno dar ligeras noticias sobre el origen de la composición"

text_texts_array = [en_text,es_text]
test_array = []
for item in text_texts_array:
    cleaned_text = clean_text(item)
    input_row = get_input_row(cleaned_text,0,text_sample_size)
    test_array.append(input_row)

test_array = standard_scaler.transform(test_array)     
Y_pred = model.predict_classes(test_array)
for id in range(len(test_array)):
    print ("Text:",text_texts_array[id][:50],"... -> Predicted lang: ", decode_langid(Y_pred[id]))
    
### END ###

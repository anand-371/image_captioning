import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
import numpy as np
import sys
import os
from PIL import Image
import pandas as pd
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle

def load_image(path, size=None):
    img = Image.open(path)

    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)

    img = img / 255.0

    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img
def process_images(data_dir,filenames, batch_size=32):
    num_images = len(filenames)
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    start_index = 0

    while start_index < num_images:
        print_progress(count=start_index, max_count=num_images)

        end_index = start_index + batch_size

        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index

        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size=img_size)
            image_batch[i] = img

        transfer_values_batch = image_model_transfer.predict(image_batch[0:current_batch_size])

        transfer_values[start_index:end_index] =transfer_values_batch[0:current_batch_size]

        start_index = end_index


    return transfer_values
def print_progress(count, max_count):
    pct_complete = count / max_count
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()       
batch_size=100
trans=[]
count=[0]
data = pd.read_csv('qwerty.csv', error_bad_lines=False, sep = '|')
filenames_train=data["image_name"]
filenames_train=filenames_train.unique()[:150]
num_images_train = len(filenames_train)
image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
img_size = K.int_shape(image_model.input)[1:3]
transfer_values_size = K.int_shape(transfer_layer.output)[1]
#run this if u want to create a .pkl file containing transfer_values
"""this data in the pkl file can be obtained when running the program without computing 
and processing through the images again"""
"""
with open('transfer_values.pkl', 'wb') as f:
    #creates transfer_values.pkl if the file does not exist
    pickle.dump(transfer_values, f)
"""
"""
transfer_values=np.asarray(process_images('flickr30k_images', filenames_train[:150]))
transfer_values.shape
"""
#loading previously saved  pickle file and storing in variable called transfer_values

with open('transfer_values.pkl', 'rb') as f:
      transfer_values = pickle.load(f)
mark_start = 'ssss '
mark_end = ' eeee'
class TokenizerWrap(Tokenizer):
    
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)

        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):

        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        
        return tokens


def mark_captions(captions_listlist):
    captions_marked = [[mark_start + str(caption) + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    
    return captions_marked   
def batch_generator(batch_size):
    while True:
        idx = np.random.randint(num_images_train,size=abatch_size)
        trans=transfer_values[idx]
        tokens = [tokens_train[i] for i in idx]
        num_tokens = [len(t) for t in tokens_train]
        max_tokens = np.max(num_tokens)
        tokens_padded = np.array(pad_sequences(tokens,maxlen=max_tokens,padding='post',truncating='post'))
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        x_data = {'decoder_input': decoder_input_data,'transfer_values_input': trans}
        y_data = {'decoder_output':decoder_output_data}
        
        yield (x_data, y_data)
captions_train=data[" comment"]
captions_train=list(captions_train.head(5*batch_size))
for i in range(len(captions_train)):
    captions_train[i]=[(captions_train[i])]
captions_train_marked = mark_captions(list(captions_train))     
def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    
    return captions_list
#captions_train_flat = flatten(captions_train_marked)
final_captions=[]
i=0
captions_train_marked
while(i<len(captions_train_marked)):
    final_captions.append(captions_train_marked[i])
    i=i+5  
final_captions=np.asarray(final_captions) 
num_words = 10000
captions_train_flat=flatten(final_captions)
np.asarray(captions_train_flat).shape    
tokenizer = TokenizerWrap(texts=captions_train_flat,num_words=num_words)
token_start = tokenizer.word_index[mark_start.strip()]    
token_end = tokenizer.word_index[mark_end.strip()]       
tokens_train = tokenizer.captions_to_tokens(final_captions)
for i in range(len(tokens_train)):
    tokens_train[i]=tokens_train[i][0] 
generator = batch_generator(batch_size=batch_size)
batch=next(generator)
batch_x =batch[0]
batch_y = batch[1]
num_captions_train = [len(captions) for captions in tokens_train]
total_num_captions_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_captions_train / batch_size)
state_size = 512
embedding_size = 128
transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')
decoder_transfer_map = Dense(state_size,activation='tanh',name='decoder_transfer_map')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,output_dim=embedding_size,name='decoder_embedding')
decoder_gru1 = GRU(state_size, name='decoder_gru1',return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',return_sequences=True)
decoder_dense = Dense(num_words,activation='linear',name='decoder_output')
def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    decoder_output = decoder_dense(net)
    
    return decoder_output
decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])    
def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean
optimizer = RMSprop(lr=1e-3)  
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))  
decoder_model.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
callback=keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)    
decoder_model.fit_generator(generator=generator,steps_per_epoch=steps_per_epoch,epochs=20)
    

import numpy as np
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
import tensorflow as tf
def batch_generator(batch_size):
    while True:
        idx=np.random.randint(total_train_size,size=batch_size)
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
def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean      
def RNN(num_words,batch_size,tokens_train_arg,transfer_values_arg,transfer_values_size,total_train_size_arg):   
	global tokens_train
	global transfer_values  
	global total_train_size
	tokens_train=tokens_train_arg
	transfer_values=transfer_values_arg
	total_train_size=total_train_size_arg
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
	decoder_output = connect_decoder(transfer_values=transfer_values_input,decoder_transfer_map=decoder_transfer_map,decoder_input=decoder_input,decoder_embedding=decoder_embedding,decoder_gru1=decoder_gru1,decoder_gru2=decoder_gru2,decoder_gru3=decoder_gru3,decoder_dense=decoder_dense)
	decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])
	return decoder_model,generator,steps_per_epoch
def connect_decoder(transfer_values,decoder_transfer_map,decoder_input,decoder_embedding,decoder_gru1,decoder_gru2,decoder_gru3,decoder_dense):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    decoder_output = decoder_dense(net)
    return decoder_output  	
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
mark_start = 'ssss '
mark_end = ' eeee'
def preprocess(data,total_train_size,num_words):
    captions_train=data[" comment"]
    captions_train=list(captions_train.head(5*total_train_size))
    for i in range(len(captions_train)):
        captions_train[i]=[(captions_train[i])]
    captions_train_marked = mark_captions(list(captions_train))     
    final_captions=[]
    i=0
    while(i<len(captions_train_marked)):
        final_captions.append(captions_train_marked[i])
        i=i+5  
    final_captions=np.asarray(final_captions)
    captions_train_flat=flatten(final_captions)
    tokenizer = TokenizerWrap(texts=captions_train_flat,num_words=num_words)
    token_start = tokenizer.word_index[mark_start.strip()]    
    token_end = tokenizer.word_index[mark_end.strip()]       
    tokens_train = tokenizer.captions_to_tokens(final_captions)
    for i in range(len(tokens_train)):
        tokens_train[i]=tokens_train[i][0] 
    return tokens_train 
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

def get_random_caption_tokens(idx):
    result = []
    for i in idx:
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)

    return result
def mark_captions(captions_listlist):
    captions_marked = [[mark_start + str(caption) + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    
    return captions_marked   
    
def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    
    return captions_list  

from keras import *
from keras.layers import *
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import KFold
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import numpy as np

HIDDEN_SIZE = 256
BATCH_SIZE = 64
EPOCH = 100
NUM_SAMPLES = 144
TRAIN_SAMPLES = 115
EMBED_DIMENSION = 300
data_path = 'dataset/100data.txt'

#build the model
def build_model(input_vocab_length,input_seq_length,output_vocab_length,hidden_size,embed_dimension,embedding_matrix):
    #encoder
	#encoder input layer
    encoder_input = Input(shape = (input_seq_length,))
	#embedding layer
    embed = Embedding(input_vocab_length, embed_dimension, weights=[embedding_matrix], input_length=input_seq_length, trainable=False)(encoder_input)
    #encoder LSTM
	encoder_LSTM = LSTM(hidden_size, return_state=True)
    _,encoder_h,encoder_c = encoder_LSTM(embed)
	#get final state of encoder
    encoder_state = [encoder_h,encoder_c]
    
    #decoder
	#decoder input layer
    decoder_input = Input(shape = (None, output_vocab_length))
    #decoder LSTM
	decoder_LSTM = LSTM(hidden_size,return_sequences=True, return_state=True)
	#input of the decoder is the final state of the encoder
    decoder_output, _, _ = decoder_LSTM(decoder_input,initial_state=encoder_state)
	#decoder dense layer
    decoder_dense = Dense(output_vocab_length,activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    
    #training
	#encoder_input[train_samples,input_seq_length] 
	#decoder_input[pos_train_samples,output_seq_length,pos_output_vocab_length] 
	#decoder_output[pos_train_samples,output_seq_length-1,pos_output_vocab_length] 
    model = Model([encoder_input,decoder_input],decoder_output)
	
	#encoder inference model
    encoder_infer = Model(encoder_input,encoder_state)
    
	#decoder inference model
    decoder_h = Input(shape=(hidden_size,))
    decoder_c = Input(shape=(hidden_size,))
	#decoder_h and decoder_c at moment t-1
    decoder_state = [decoder_h, decoder_c]
    decoder_infer_output, decoder_infer_h, decoder_infer_c = decoder_LSTM(decoder_input,initial_state=decoder_state)
    #decoder_infer_h and decoder_infer_c at moment t
	decoder_infer_state = [decoder_infer_h, decoder_infer_c]
    decoder_infer_output = decoder_dense(decoder_infer_output)
	#moment (t-1,t)
    decoder_infer = Model([decoder_input]+decoder_state,[decoder_infer_output]+decoder_infer_state)
    
    return model, encoder_infer, decoder_infer

#method of getting vocabulary list 
def get_words(text):
    words = []
    for i in text:
        line = i.split()
        for j in line:
            words.append(j)  
    words = sorted(list(set(words)))
    return words
	
#get data
df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:,]
df.columns=['inputs','targets']
#add <STA> and <EOS> to target data
df['targets'] = df['targets'].apply(lambda x: '<STA> '+x+' <EOS>')

input_texts_all = df.inputs.values.tolist()
target_texts_all = df.targets.values.tolist()

#get vocabulary list
input_words = get_words(input_texts_all)
target_words = get_words(target_texts_all)

#add <UNK> to the vocabulary list 
input_words.append('<UNK>')
input_words = sorted(list(input_words))
target_words.append('<UNK>')
target_words = sorted(list(target_words))

#get the training text: the first 80% data
input_texts= [input_texts_all[i:i+TRAIN_SAMPLES] for i in range(0, len(input_texts_all), TRAIN_SAMPLES)][0]
target_texts= [target_texts_all[i:i+TRAIN_SAMPLES] for i in range(0, len(target_texts_all), TRAIN_SAMPLES)][0]

INPUT_SEQ_LENGTH = max([len(i.split()) for i in input_texts])
OUTPUT_SEQ_LENGTH = max([len(i.split()) for i in target_texts])
INPUT_VOCAB_LENGTH = len(input_words)
OUTPUT_VOCAB_LENGTH = len(target_words)

#build empty encoder_input, decoder_input, decoder_output
encoder_input = np.zeros((TRAIN_SAMPLES,INPUT_SEQ_LENGTH),dtype = int)
decoder_input = np.zeros((TRAIN_SAMPLES,OUTPUT_SEQ_LENGTH,OUTPUT_VOCAB_LENGTH))
decoder_output = np.zeros((TRAIN_SAMPLES,OUTPUT_SEQ_LENGTH,OUTPUT_VOCAB_LENGTH))

#build word vocabulary dictionary
input_dict = {word:index for index,word in enumerate(input_words)}
input_dict_reverse = {index:word for index,word in enumerate(input_words)}
target_dict = {word:index for index,word in enumerate(target_words)}
target_dict_reverse = {index:word for index,word in enumerate(target_words)}

#update encoder_input, decoder_input, decoder_output
for seq_index,seq in enumerate(input_texts):
    for word_index,word in enumerate(seq.split()):
        if word not in input_dict:
            word = '<UNK>'
        encoder_input[seq_index,word_index] = input_dict[word]
        
for seq_index,seq in enumerate(target_texts):
    for word_index,word in enumerate(seq.split()):
        if word not in target_dict:
            word = '<UNK>'
        decoder_input[seq_index,word_index,target_dict[word]] = 1.0
        if word_index > 0:
            decoder_output[seq_index,word_index-1,target_dict[word]] = 1.0

# embedding
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.42B.300d/output-20k.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))#key:word  value:embed

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((INPUT_VOCAB_LENGTH, 300))
for i,word in enumerate(input_words):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#build the model
model_train, encoder_infer, decoder_infer = build_model(INPUT_VOCAB_LENGTH, INPUT_SEQ_LENGTH, OUTPUT_VOCAB_LENGTH, HIDDEN_SIZE, EMBED_DIMENSION, embedding_matrix)
model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model_train.summary()
encoder_infer.summary()
decoder_infer.summary()

#use k-fold cross validation
kfold = KFold(5, shuffle=False)
scores = []

#train the model
for train, test in kfold.split(encoder_input, decoder_input, decoder_output):
    model_train.fit([encoder_input[train],decoder_input[train]],decoder_output[train],batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.2)
    model_train.evaluate([encoder_input[test],decoder_input[test]],decoder_output[test])

#save the model
model_train.save('seq2seq_keras-master/COMP393_model.h5')
encoder_infer.save('seq2seq_keras-master/COMP393_encoder_infer.h5')
decoder_infer.save('seq2seq_keras-master/COMP393_decoder_infer.h5')

#Evaluation

def predict_paraphrases(source,encoder_infer,decoder_infer,output_seq_length,output_vocab_length):
    #get the state of the source
	state = encoder_infer.predict(source)
    predict_seq = np.zeros((1,1,output_vocab_length))
    predict_seq[0,0,target_dict['<STA>']] = 1

    output = ''
	#use the last predicted word as input to predict the next word until the terminator is predicted
    for i in range(output_seq_length):
        yhat,h,c = decoder_infer.predict([predict_seq]+state)
        word_index = np.argmax(yhat[0,-1,:])
        word = target_dict_reverse[word_index]
        output = output+word+" "
        state = [h,c]
        predict_seq = np.zeros((1,1,output_vocab_length))
        predict_seq[0,0,word_index] = 1
        if word == '<EOS>':
            break
    return output

#load WMD model
model = KeyedVectors.load_word2vec_format(
    'Google-News-vectors-negative300.bin', binary=True, limit=90000)
model.init_sims(replace=True) 

#get evaluation text
test_data_path = 'dataset/4000data.txt'
df_test = pd.read_table(test_data_path,header=None).iloc[3260:,:,]
df_test.columns=['inputs','targets']
test_input_texts = df_test.inputs.values.tolist()
test_target_texts = df_test.targets.values.tolist()
#get testing text
df_test = pd.read_table(data_path,header=None).iloc[115:144,:,]
df_test.columns=['inputs','targets']
testing_text = df_test.inputs.values.tolist()
#empty list to record max bleu and min WMD score for each noun-modifier
bleu_score_list = []
wmd_score_list = []

#start testing
for i in range(0,len(testing_text)):
    encoder_input_test = np.zeros((1,INPUT_SEQ_LENGTH), dtype = int)
    seq=testing_text[i]
    #predict the paraphrase
    for word_index, word in enumerate(seq.split()):
        if word not in input_dict:
            word = '<UNK>'
        encoder_input_test[0,word_index] = input_dict[word]
    test = encoder_input_test[0:1,]
	#get the generated paraphrase
    out = predict_paraphrases(test,encoder_infer,decoder_infer,OUTPUT_SEQ_LENGTH,OUTPUT_VOCAB_LENGTH)
	
	#build two empty lists
    bleu_sequence_score = []
	wmd_sequence_score = []
    candidate = out.split()
    candidate.pop()
    for j in range(0,len(test_input_texts)):
        if seq == test_input_texts[j]:
			#new an empty list
            reference = []
            reference.append(test_target_texts[j].split())
			#cal bleu
            bleu_score = sentence_bleu(reference, candidate)
            bleu_sequence_score.append(bleu_score)
			#cal WMD
			wmd_score = model.wmdistance(reference[0], candidate)
            wmd_sequence_score.append(wmd_score)
			#print
            print(seq)
            print(candidate)
            print(reference)
			print(wmd_score)
            print(bleu_score)
            print('\n')
	#max bleu score
    max_score = max(bleu_sequence_score)
    bleu_score_list.append(max_score)
	#min wmd score
	min_score = min(wmd_sequence_score)
    wmd_score_list.append(min_score)

#calculate bleu average score
bleu_average_score = sum(bleu_score_list)/len(bleu_score_list) 
print(bleu_average_score)

#calculate wmd average score
#remove inf
result = []
result = [c for c in wmd_score_list if c != float("inf")]
wmd_average_score = sum(result)/len(result) 
print(wmd_average_score)


import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle as pkl
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

# Define path for file with datasets
dataset = 'Datasets\VGCoST_VideoGameDialogue_Corpus\ENG\Portal_merged.txt'

# Read the data
with open(dataset, encoding='ISO-8859-1') as f:
    data = f.read()

# Remove unwanted characters using regex
data = re.sub(r"[\"']", "", data)

# Convert to lower case and save as a list
corpus = data.lower().split("\n")

# Define tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Define n_gram_seqs function
def n_gram_seqs(corpus, tokenizer):
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			# Generate subphrase
			n_gram_sequence = token_list[:i+1]
			# Append subphrase to input_sequences list
			input_sequences.append(n_gram_sequence)
	return input_sequences

# Apply the n_gram_seqs transformation to the whole corpus
input_sequences = n_gram_seqs(corpus, tokenizer)

# Save max length
max_sequence_len = max([len(x) for x in input_sequences])

def pad_seqs(input_sequences, maxlen):
    padded_sequences = pad_sequences(input_sequences, maxlen=maxlen, padding='pre')
    return padded_sequences

# Pad the whole corpus
input_sequences = pad_seqs(input_sequences, max_sequence_len)

def features_and_labels(input_sequences, total_words):
    features = input_sequences[:,:-1]
    labels = input_sequences[:,-1]
    one_hot_labels = to_categorical(labels, num_classes=total_words)
    return features, one_hot_labels

# Split the whole corpus
features, labels = features_and_labels(input_sequences, total_words)

def create_model(total_words, max_sequence_len):

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

tf.compat.v1.disable_eager_execution()
   
model = create_model(total_words, max_sequence_len)

# Train the model
model.fit(features, labels, epochs=50, verbose=0)

# Save the model
model.save('model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pkl.dump(tokenizer, handle)


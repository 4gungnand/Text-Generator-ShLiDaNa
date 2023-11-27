import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

def text_generator(input_text):
    # Load the model
    model = load_model('model.h5')

    # Take user input
    seed_text = input("Enter your seed text:\n")
    next_words = 20
    max_sequence_len = 21 # Based on training data (refer to ipynb file)

    for _ in range(next_words):
        # Initialize tokenizer
        tokenizer = Tokenizer()
        # Convert the text into sequences
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # Get the probabilities of predicting a word
        predicted = model.predict(token_list, verbose=0)
        # Choose the next word based on the maximum probability
        predicted = np.argmax(predicted, axis=-1).item()
        # Get the actual word from the word index
        output_word = tokenizer.index_word[predicted]
        # Append to the current text
        seed_text += " " + output_word

    return seed_text

def __main__():
    st.title("Text Generation")
    st.write("This app generates text based on the input text.")
    input_text = st.text_input("Enter your seed text:")
    if st.button("Generate"):
        output_text = text_generator(input_text)
        st.write(output_text)
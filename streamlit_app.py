import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

# Load the model
model, tokenizer = load_model_and_tokenizer()

def text_generator(input_text, next_words=20, model=model, tokenizer=tokenizer):
    seed_text = input_text
    for _ in range(next_words):
        # Convert the text into sequences
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequences
        token_list = pad_sequences([token_list], maxlen=20, padding='pre')
        # Get the probabilities of predicting a word
        predicted = model.predict(token_list, verbose=0)
        # Choose the next word based on the maximum probability
        predicted = np.argmax(predicted, axis=-1).item()
        # Get the actual word from the word index
        output_word = tokenizer.index_word[predicted]
        # Append to the current text
        seed_text += " " + output_word

    return seed_text


st.text("Halo dek")
    
st.title("Text Generation")

st.write("This app generates text based on the input text.")

input_text = st.text_input("Enter your seed text:")

next_words = st.slider("How many words do you want to generate?", 1, 15)

if st.button("Generate"):
    output_text = text_generator(input_text, next_words, model, tokenizer)
    st.write(output_text)
    
    

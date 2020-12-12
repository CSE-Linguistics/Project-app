import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from .cleaner_single import text_strip_single
from tensorflow.keras.models import Model
import pickle
import os
"""
MODEL_PATH
   |------ lstm_model
   |        |---------- assets
   |        |---------- variables
   |        |---------- saved_model.pb
   |------ x_tokenizer.pickle
   |------ y_tokenizer.pickle
"""
MODEL_PATH = "/home/sumanyu/app_nlp_project/app/sumapp/app_form/models/seq-to-seq" # Enter the path to the model on your local device
# Load model from file
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'lstm-model'))
print(model.summary())
def get_summary_abstractive(Text:str):
    """
    The given function builds the decoder after sharing some weights with the encoder.
    After building the decoder, this would generate the summary.
    Arguments:
    Text: str
    The text to be summarized
    Output:
    str
    The summary
    """
    # Load both the tokenizers from files
    with open(os.path.join(MODEL_PATH,'x_tokenizer.pickle'), 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    
    with open(os.path.join(MODEL_PATH,'y_tokenizer.pickle'), 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    # Various hyperparameters for the Model
    max_text_len    = 100
    max_summary_len = 15
    latent_dim      = 300
    embedding_dim   = 200

    # Get references to the layers in the saved model
    encoder_inputs = model.input[0]
    decoder_inputs = model.input[1] 
    dec_emb_layer  = model.layers[5]
    decoder_lstm   = model.layers[7]
    attn_layer     = model.layers[8]
    Concatenate    = model.layers[9]
    decoder_dense  = model.layers[10]

    encoder_outputs, state_h, state_c = model.layers[6].output # lstm_1

    # Encode input sequence to get the context vector
    encoder_model = Model(inputs = encoder_inputs, outputs = [encoder_outputs, state_h, state_c])

    # Setup decoder: c, h hold context
    decoder_state_input_h      = Input(shape = (latent_dim,),name = "dec_state_inp_h")
    decoder_state_input_c      = Input(shape = (latent_dim,), name= "dec_state_inp_c")
    decoder_hidden_states      = Input(shape = (max_text_len, latent_dim),name ="dec_hidden_state")

    dec_emb = dec_emb_layer(decoder_inputs) 
    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state = [decoder_state_input_h, decoder_state_input_c])

    # attention layers
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_states, decoder_outputs])
    decoder_inf_concat            = Concatenate([decoder_outputs, attn_out_inf])
    decoder_outputs2              = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_states, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h, state_c])

    #Dictionary for index to word
    target_index_word = y_tokenizer.index_word
    source_index_word = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    def decode_sequence(input_seq):
        """
        INTERNAL FUNCTION. NOT TO BE USED DIRECTLY!

        The given function would summarize a sequence of tokenized words

        input:
        input_seq: Tokenized words
        returns:
        The raw summary
        """
        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first word of target sequence with the start word('sostok').
        target_seq[0, 0] = target_word_index['sostok']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
    
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word        = target_index_word[sampled_token_index]
            
            if sampled_word != 'eostok':
                decoded_sentence += ' ' + sampled_word

            # Exit condition: either hit max length 
            # or find stop word.
            if sampled_word == 'eostok'  or \
                len(decoded_sentence.split()) >= max_summary_len-1:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            e_h, e_c = h, c

        return decoded_sentence

    print("Text:", Text)
    Text    =   pad_sequences(x_tokenizer.texts_to_sequences([text_strip_single(Text)]), \
                                    maxlen = max_text_len, padding = 'post')
    decoded_text = decode_sequence(Text.reshape(1, max_text_len))[6:-4]
    print("Predicted Summary:", decoded_text)

    return decoded_text
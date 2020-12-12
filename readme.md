# A summary of Summarizers

This is an app consisting of different summarizers that we have built/used as a part of our project for CS 626.

The intent of this app is for people to be able to understand the different summarizers and make a comparative study of their own to see which works for them the best for different situations.


## Summarizers Used:
1 - BertSum Based summarizer: Credits -> https://arxiv.org/abs/1906.04165

2 - BART Summarizer: Credits -> https://huggingface.co/transformers/model_doc/bart.html

3 - Pegasus Summarizer: Credits -> https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html

4 - Our own Sequence to Sequence Abstractive Summarizer using LSTMs and attention

## Steps for setting up the app for use:
1 - Install sentencepiece tokenzier for pegasus: https://github.com/google/sentencepiece

2 - pip install requirements.txt

3 - Setting up the seqtoseq model: place the models directory somewhere in your local device, and edit the path in app_form/seq2seq_app_submission.py. Can be downloaded from https://drive.google.com/drive/folders/1kfznlcbNxITOrG0YZCVHUPg8_B4HNLwi?usp=sharing

4 - python3 manage.py runserver
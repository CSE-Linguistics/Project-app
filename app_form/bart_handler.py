from transformers import AutoTokenizer, AutoModel, pipeline
import torch
def summary_bart(sentence:str):
    """
    This would generate the summary using the pretrained BART summarizer

    arguments:
    sentence: str 
    The sentences that need to be summarized

    output:
    the summary as an str
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModel.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline(task="summarization")
    summary = summarizer(sentence,min_length= 6, max_length = max(len(sentence.split(" "))//2, 60))
    return summary[0]['summary_text']
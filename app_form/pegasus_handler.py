from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
import torch

def summary_pegasus(sentence:str):
    """
    This would generate the summary using the pretrained pegasus abstractive summarizer

    arguments:
    sentence: str 
    The sentences that need to be summarized

    output:
    the summary as an str
    """
    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    summarizer = pipeline(task="summarization",model = model, tokenizer = tokenizer)
    summary = summarizer(sentence)
    return summary[0]['summary_text']

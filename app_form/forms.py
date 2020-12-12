from django import forms
from .choices import MODEL_CHOICES

class AppForm(forms.Form):
    """
    The final form that needs to be generated.
    
    There are four options:

    1 -> Via BART
    
    2 -> Via BERTSum + k-means
    
    3 -> via pegasus
    
    4 -> Via our own Sequence to Sequence Summarizer
    """
    full_text = forms.CharField(widget = forms.Textarea)
    model_needed = forms.ChoiceField(choices = MODEL_CHOICES)
    pass

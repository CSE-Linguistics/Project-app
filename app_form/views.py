"This file has the view needed for the form"
from django.shortcuts import render
from .forms import AppForm
from .bart_handler import summary_bart
from .pegasus_handler import summary_pegasus
# Create your views here.
from summarizer import Summarizer

summ = Summarizer()
from .seq2seq_app_submission import get_summary_abstractive
def transformForm(request):
    """
    The django view that needs to be generated for the form.
    """
    input_text = ""
    output_text = ""
    if request.method == "POST":
        form = AppForm(request.POST)
        if form.is_valid():
            print("Reaching here?")
            input_text = form.cleaned_data['full_text']
            model_choice = form.cleaned_data['model_needed']
            if len(input_text.strip().split(" ")) < 20:
                output_text = "Do you really need summarization?"
            elif model_choice == '1':
                output_text = summ(input_text) #This summarizer is the BERTSum + K-Means classification
            elif model_choice == "2":
                output_text = summary_bart(input_text)
            elif model_choice == "3":
                output_text = summary_pegasus(input_text)
            elif model_choice =="4":
                output_text = get_summary_abstractive(input_text)
        else:
            form = AppForm()
    else:
        form = AppForm()
    
    return render(request, 'form.html', {"form":form, "otpt":output_text })

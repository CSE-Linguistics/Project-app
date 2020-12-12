"""This file has the function needed to clean up sentences so that the words in the sentences can be used to be tokenized"""
import re

#Removes non-alphabetic characters:
def text_strip_single(row):
    """
    Regex based text strip for a sentence


    Arguments:
    ----------
    row: str
        a sentence

    Returns:
    --------
    str
        Cleaned up sentence fit for tokenization

    This has been taken from: https://www.kaggle.com/sandeepbhogaraju/text-summarization-with-seq2seq-model
    """
        
    #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
    
    row=re.sub("(\\t)", ' ', str(row)).lower() #remove escape charecters
    row=re.sub("(\\r)", ' ', str(row)).lower() 
    row=re.sub("(\\n)", ' ', str(row)).lower()
    
    row=re.sub("(__+)", ' ', str(row)).lower()   #remove _ if it occors more than one time consecutively
    row=re.sub("(--+)", ' ', str(row)).lower()   #remove - if it occors more than one time consecutively
    row=re.sub("(~~+)", ' ', str(row)).lower()   #remove ~ if it occors more than one time consecutively
    row=re.sub("(\+\++)", ' ', str(row)).lower()   #remove + if it occors more than one time consecutively
    row=re.sub("(\.\.+)", ' ', str(row)).lower()   #remove . if it occors more than one time consecutively
    
    row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!
    
    row=re.sub("(mailto:)", ' ', str(row)).lower() #remove mailto:
    row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() #remove \x9* in text
    row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() #replace INC nums to INC_NUM
    row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() #replace CM# and CHG# to CM_NUM
    
    
    row=re.sub("(\.\s+)", ' ', str(row)).lower() #remove full stop at end of words(not between)
    row=re.sub("(\-\s+)", ' ', str(row)).lower() #remove - at end of words(not between)
    row=re.sub("(\:\s+)", ' ', str(row)).lower() #remove : at end of words(not between)
    
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
    
    #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
        repl_url = url.group(3)
        row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
    except:
        pass #there might be emails with no url in them
    

    
    row = re.sub("(\s+)",' ',str(row)).lower() #remove multiple spaces
    
    #Should always be last
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
    return row
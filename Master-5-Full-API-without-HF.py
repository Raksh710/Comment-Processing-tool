# importing the required packages and functions
import PyPDF2, numpy as np, pandas as pd # NOTE: Use the version 2.12.1 of PyPDF2 as the newer versions have deprecated many functionalities
from PyPDF2 import PdfFileReader
import re
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel, pipeline , TokenClassificationPipeline , AutoModelForTokenClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import os
from transformers.pipelines import AggregationStrategy
from flask import Flask, request, render_template, jsonify


my_dir = 'C:\\Users\\rsinha\\Desktop\\Comment Tool\\OneDrive_1_2-10-2023\\'
# Creating a "cleaning" function which we'll use to clean comments
def cleaner(x):
    x = str(x)
    x = re.sub(r'\[[0-9]*\]',' ',x)
    x = re.sub(r'\s+',' ',x)
    x = x.lower()
    x = re.sub(r'\d',' ',x)
    x = re.sub(r'\s+',' ',x)
    return x

# Creating a function to convert the contents of a pdf file into a list
def pdf_to_text(pdf):
    f = open(pdf,mode='rb')
    pdf_text = []

    pdf_reader3 = PyPDF2.PdfFileReader(f)

    for p in range(pdf_reader3.numPages):
        page = pdf_reader3.getPage(p)
        pdf_text.append(page.extractText())

    f.close()
    all_text = ' '.join(pdf_text).replace('\n','')
    cleaned_text = cleaner(all_text).split(' re: ')[-1]
    
    return cleaned_text

# Creating an embedding function (basically vectorizing every word in every sentence in order to grab granular details present in each sentence)
def give_embed(sentences):
    def mean_pooling(model_output, attention_mask):

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    #print("Sentence embeddings:")
    return sentence_embeddings

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])
    
# Creating a keyword extracting function
def key_word_extractor(x):
    try:
        if len(x) > 0:
            return extractor(x)
        else:
            return np.nan
    except:
        return np.nan
    
# Creating a function to bin a compound polarity score into 9 categories namely: Extremely Negative, Very Negative, Negative, Neutral-Negative, Neutral, Neutral-Positive, Positive, Very Positive, Extremely Positive
def final_sent_calc(x):
    
    def estimator(a):
        return float(str(a)[:6])
    def estimator2(b):
        return float(str(b)[:7])

    if x in np.asarray(pd.Series(np.arange(-1.001, -0.778,0.0001)).apply(lambda x: estimator2(x))):
        return 'Extremely Negative'
    elif x in np.asarray(pd.Series(np.arange(-0.778, -0.556,0.0001)).apply(lambda x: estimator2(x))):
        return 'Very Negative'
    elif x in np.asarray(pd.Series(np.arange(-0.556, -0.333,0.0001)).apply(lambda x: estimator2(x))):
        return 'Negative'
    elif x in np.asarray(pd.Series(np.arange(-0.333, -0.111,0.0001)).apply(lambda x: estimator2(x))):
        return 'Neutral-Negative'
    elif x in np.asarray(pd.Series(np.arange(-0.111, 0.000,0.0001)).apply(lambda x: estimator2(x))):
        return 'Neutral'
    elif x in np.asarray(pd.Series(np.arange(0, 0.111,0.0001)).apply(lambda x: estimator(x))):
        return 'Neutral'
    elif x in np.asarray(pd.Series(np.arange(0.111, 0.333,0.0001)).apply(lambda x: estimator(x))):
        return 'Neutral-Positive'
    elif x in np.asarray(pd.Series(np.arange(0.333, 0.556,0.0001)).apply(lambda x: estimator(x))):
        return 'Positive'
    elif x in np.asarray(pd.Series(np.arange(0.556, 0.778,0.0001)).apply(lambda x: estimator(x))):
        return 'Very Positive'
    elif x in np.asarray(pd.Series(np.arange(0.778, 1.001,0.0001)).apply(lambda x: estimator(x))):
        return 'Extremely Positive'
    
def summary_func(x):
    word_count = int(len(x.split()))
    limit = 500
    if word_count > limit:
        return summarizer(' '.join(x.split()[:limit]))[0]['summary_text']
    else:
        return summarizer(x)[0]['summary_text']
    
    
app = Flask(__name__)

@app.route('/cpt')
def comment_process_tool():
    my_dir = 'C:\\Users\\rsinha\\Desktop\\Comment Tool\\OneDrive_1_2-10-2023\\'
    pdfs = []
    for i in os.listdir(my_dir):
        if '.pdf' in i:
            pdfs.append(i)
    
    text_of_pdfs = []
    for i in pdfs:
        text_of_pdfs.append(pdf_to_text(i))

    df_comments = pd.DataFrame({'Comment':text_of_pdfs}) 
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # using tokenizer from pretrained model
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # Assembling the model with pre-trained weights
    
    l1 = df_comments['Comment'].tolist() # maing a list of all the comments
    # Ensuring that all the comments belong to string datatype
    l2 = []
    for i in l1:
        l2.append(str(i))
        
    series_embeds = give_embed(l2) # Applying our embedding function on our corpus
    # Evaluating the similarity for each combination of comments using the cosine similarity method 
    sent1 = []
    sent2 = []
    similarity = []

    for i in range(len(l2)-1):
        #sent1.append(l2[i])
        for j in range(i+1, len(l2)):
            sent1.append(l2[i])
            sent2.append(l2[j])
            similarity.append(cosine_similarity(series_embeds[i].reshape(1,-1) , series_embeds[j].reshape(1,-1) )[0][0])
            
    # creating the dataframe for all combination of comments along with their similarity scores
    df_sentence_similarity = pd.DataFrame({'Docuemnt1':sent1, 'Document2': sent2,'Cosine_Similarity_Score': similarity}) 
    
    # Importing the model and extractor from huggingface.co
    model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
    extractor = KeyphraseExtractionPipeline(model=model_name)
    
    df_comments['keywords'] = df_comments['Comment'].apply(lambda x: key_word_extractor(x)) # Extracting out the keywords from each comment
    
    # Creating a list which contains the number of keywords of each comment
    key_cnt = []
    for i in range(len(df_comments)):
        try:
            key_cnt.append(len(df_comments['keywords'][i]))
        except:
            key_cnt.append(0)
            
    df_comments['key_count'] = key_cnt # Creating a new column which contains the number of keywords in each comment
    
    rev_len = [len(x.split()) if x is not np.nan else 0 for x in df_comments['Comment']]  # Creating a list which contains the number of words in each comment
    df_comments['review_length'] = rev_len # Creating a new column which contains the number of words in each comment
    
    df_comments['proportion_keywords'] = 100 * df_comments['key_count']/df_comments['review_length'] # proportion of key words present in each comment depending on the context of each comment
    
    sia = SentimentIntensityAnalyzer() # Creating an instance of SentimentIntensityAnalyzer function to analyze the sentiment of each comment by calling its polarity_scores method
    
    df_comments['sentiment_score'] = df_comments['Comment'].apply(lambda x: sia.polarity_scores(x)['compound']) # Applying polarity score method to get the compound polarity score of each comment
    df_comments['sentiment'] = df_comments['sentiment_score'].apply(lambda x: final_sent_calc(x)) # Evaluating the sentiment type by applying the binning function
    
    #from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    df_comments['summary'] = df_comments['Comment'].apply(lambda x: summary_func(x))
    
    print('DONE')
    return jsonify(df_comments.to_json()) , jsonify(df_sentence_similarity.to_json())



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
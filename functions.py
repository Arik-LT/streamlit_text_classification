import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop= stopwords.words('english')

def cleaning_text(text):
    text=text.lower()
    obj=re.compile(r"<.*?>")                     #removing html tags
    text=obj.sub(r" ",text)
    obj=re.compile(r"https://\S+|http://\S+")    #removing url
    text=obj.sub(r" ",text)
    obj=re.compile(r"[^\w\s]")                   #removing punctuations
    text=obj.sub(r" ",text)
    obj=re.compile(r"\d{1,}")                    #removing digits
    text=obj.sub(r" ",text)
    obj=re.compile(r"_+")                        #removing underscore
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s\w\s")                    #removing single character
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s{2,}")                    #removing multiple spaces
    text=obj.sub(r" ",text)
       
    #stemmer = SnowballStemmer("english")
    #text=[stemmer.stem(word) for word in text.split() if word not in stop]
    text=[word for word in text.split() if word not in stop]
    return " ".join(text)

def removing_single_letters(text):
    tmp = re.sub(r'\b\w\b', ' ', text)
    return re.sub(r'\s{2,}', ' ', tmp).strip()
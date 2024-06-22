from linecache import clearcache
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
import joblib
import re
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import urllib.request as urllib
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
import time 
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service  # Import Service
from selenium.webdriver.chrome.options import Options  # Import Options

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

def returnytcomments(url):
    data=[]

    chrome_options = Options()
    chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # Update with the correct path
    driver_path = r'C:\Users\Gurmehar Singh\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'  # Correct path to the chromedriver executable

    service = Service(executable_path=driver_path)  # Initialize Service with the path

    with Chrome(service=service, options=chrome_options) as driver:  # Pass Service to Chrome
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        for item in range(5):
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(2)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
            data.append(comment.text)
    
    return data

def clean(org_comments):
    y = []
    for x in org_comments:
        x = x.split()
        x = [i.lower().strip() for i in x]
        x = [i for i in x if i not in stop_words]
        x = [i for i in x if len(i) > 2]
        x = [wnl.lemmatize(i) for i in x]
        y.append(' '.join(x))
    return y

def create_wordcloud(clean_reviews):
    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wcstops, background_color='white').generate(for_wc)
    plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic')
    plt.axis('off')
    plt.tight_layout()
    # clearcache(directory='static/images') - check this!
    plt.savefig('woc.png')
    plt.close()

def returnsentiment(x):
    score = sia.polarity_scores(x)['compound']

    if score > 0:
        sent = 'Positive'
    elif score == 0:
        sent = 'Negative'
    else:
        sent = 'Neutral'
    return score, sent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')

    org_comments = returnytcomments(url)

    tmp = []

    for i in org_comments:
        if 5 < len(i) <= 500:
            tmp.append(i)

    org_comments = tmp

    clean_comments = clean(org_comments)

    create_wordcloud(clean_comments)

    np, nn, nne = 0, 0, 0

    preds = []
    srs = []
    dic = []

    for i in clean_comments:
        score, sent = returnsentiment(i)
        srs.append(score)
        if sent == 'Positive':
            preds.append('POSITIVE')
            np += 1
        elif sent == 'Negative':
            preds.append('NEGATIVE')
            nn += 1
        else:
            preds.append('NEUTRAL')
            nne += 1

    for i, cc in enumerate(clean_comments):
        x = {}
        x['sent'] = preds[i]
        x['clean_comment'] = cc
        x['org_comment'] = org_comments[i]
        x['score'] = srs[i]
        dic.append(x)

    return render_template('result.html', n=len(clean_comments), nn=nn, np=np, nne=nne, dic=dic)

@app.route('/cloud')
def cloud():
    return render_template('cloud.html')

class CleanCache:
    # For clearing any residual files present in the cache
    def __init__(self, directory=None):
        self.clean_path = directory
        
        if os.listdir(self.clean_path) != list():
            # Remove the files
            files = os.listdir(self.clean_path)
            for filename in files:
                print(filename)
                os.remove(os.path.join(self.clean_path, filename))
        print("Cleaned!")

if __name__ == '__main__':
    app.run(debug=True)
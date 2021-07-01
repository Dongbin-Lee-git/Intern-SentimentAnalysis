# from konlpy.tag import Mecab
from eunjeon import Mecab
from tqdm import tqdm
import re
import pickle
import csv
import pandas as pd

def clean_text(text):
    text = text.replace(".", " ").strip()
    text = text.replace("·", " ").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
    text = re.sub(pattern=pattern, repl='',string=text)
    return text

def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    nouns = [s for s, t in tagged if t in ['SL', 'NNG', 'NNP'] and len(s) > 1]
    return nouns

def tokenize(df):
    tokenizer = Mecab()
    processed_data = []
    for sent in tqdm(df['description']):
        sentence = clean_text(sent.replace('\n', '').strip())
        processed_data.append(get_nouns(tokenizer, sentence))
    return processed_data

def save_processed_data(processed_data):
    with open('tosel_no_ad.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)


if __name__ == '__main__':
    df = pd.read_csv('tosel_no_ad.txt', sep='\n', header = None)
    df.columns = ['description']
    print(df.head(10))
    processed_data = tokenize(df)
    save_processed_data(processed_data)
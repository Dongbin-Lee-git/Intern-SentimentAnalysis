import re
import string

import pandas as pd
data = pd.read_excel('sampledata.xlsx')
data.columns = ['Text']
print(data.head())
print(data.shape)


'''
Clean data
'''
from eunjeon import Mecab
from konlpy.tag import Okt

def preprocword(text):
    def clean_text(text):
        text = text.replace(".", " ").strip()
        text = text.replace("·", " ").strip()
        pattern = '[a-zA-Z0-9]'
        text = re.sub(pattern=pattern, repl='', string=text)
        pattern = '[-=+,#/\:$.@*\"※&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》▲▶△“’_♥■]'
        text = re.sub(pattern=pattern, repl='', string=text)
        return text

    def delete(keyword):
        keyword = deleteW(keyword, "!")
        keyword = deleteW(keyword, "?")
        keyword = deleteW(keyword, "!?")
        keyword = deleteW(keyword, "?!")
        keyword = deleteW(keyword, ";")
        keyword = deleteW(keyword, "~")
        keyword = dltdot(keyword)
        keyword = clean_text(keyword)
        return keyword

    def deleteW(keyword, delword):
        while 1:
            if delword + delword in keyword:
                # print("변경 전: " + keyword)
                keyword = keyword.replace(delword + delword, delword)
                # print("변경 후: " + keyword)
            else:
                break;
        return keyword

    def dltdot(keyword):
        while 1:
            if "…" in keyword:
                # print("변경 전: " + keyword)
                keyword = keyword.replace("…", "..")
                # print("변경 후: " + keyword)
            else:
                break;

        while 1:
            if "..." in keyword:
                # print("변경 전: " + keyword)
                keyword = keyword.replace("...", "..")
                # print("변경 후: " + keyword)
            else:
                break;
        return keyword

    keyword = text  # <-원문 넣을 곳
    mecab = Okt()
    keyword = delete(str(keyword))
    return mecab.morphs(keyword)

data['tokens'] = data['Text'].apply(lambda x: preprocword(x))
data['Text_Final'] = [' '.join(sen) for sen in data['tokens']]

data = data[['Text_Final', 'tokens']]
print(data.head(10))
#
# def remove_punct(text):
#     text_nopunct = ''
#     text_nopunct = re.sub('['+string.punctuation+']', '', text)
#     return text_nopunct
#
# data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))

#from nltk import word_tokenize, WordNetLemmatizer

''' 
split data into test and train
'''

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

all_test_words = [word for tokens in data["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print(len(TEST_VOCAB))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))


MAX_SEQUENCE_LENGTH = max(test_sentence_lengths)
tokenizer = Tokenizer(num_words=len(TEST_VOCAB))
tokenizer.fit_on_texts(data["Text_Final"].tolist())
test_sequences = tokenizer.texts_to_sequences(data["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
for i in test_cnn_data:
    print(i)
'''
Test CNN
'''

model = load_model("best_model10.h5")
predictions = model.predict(test_cnn_data)
labels = [1,0]
predicton_labels=[]
for p in predictions:
    if np.argmax(p) == 1:
        print("긍정")
    else:
        print("부정")
# print(sum(data.Label == predicton_labels)/len(predicton_labels))
#
# print(data.Label.value_counts())
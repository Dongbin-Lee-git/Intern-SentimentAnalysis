import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm_pandas, tqdm
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_excel('naver_steam_hyundai_shop.xlsx')
data.columns = ['reviews', 'label']
# data = pd.read_excel('sampledata.xlsx')
# data.columns = ['reviews']
print(data.head())
print(data.label.unique())
print(data.shape)
data['reviews'].replace('', np.nan, inplace=True)
data.drop_duplicates(subset=['reviews'], inplace=True)
print("총 샘플의 수 : ", len(data))
data.dropna(axis=0)
#train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data = data
print('훈련용 : ', len(train_data))
#print('테스트용 : ', len(test_data))
okt = Okt()

import re

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
    keyword = delete(str(keyword))
    text = okt.morphs(keyword)
    stopwords = ['블루핸즈','블루링크','블루', '핸즈','링크','도', '는', '다', '의', '가',
             '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네',
             '1','2','3','4','5','6','7','8','9','0','들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

    text = [word for word in text if not word in stopwords]
    return text

train_data['tokenized'] = np.load('X_save.npy', allow_pickle=True)

# tqdm.pandas()
# train_data['tokenized'] = train_data['reviews'].progress_apply(lambda x: preprocword(x))
#test_data['tokenized'] = test_data['reviews'].progress_apply(lambda x: preprocword(x))

negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)

negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)
print(positive_word_count.most_common(20))

text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
print('부정 리뷰의 평균 길이 :', np.mean(text_len))


all_training_words = [word for tokens in train_data["tokenized"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in train_data["tokenized"]]
train_data['Text_Final'] = [' '.join(sen) for sen in train_data["tokenized"]]

TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
from gensim import models
w2v = models.KeyedVectors.load("steam_hyundai_naver_shop_w2v")

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=100):
    if len(tokens_list) <1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokenized'].apply(lambda x: get_average_word2vec(x,vectors,generate_missing=generate_missing))
    return list(embeddings)

'''
get embeddings
'''

training_embeddings = get_word2vec_embeddings(w2v, train_data, generate_missing=True)
MAX_SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 100

'''
Tokenize and Pad sequences
'''
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(train_data["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(train_data["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print("found %s unique tokens. " % len(train_word_index))

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embeddings_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word, index in train_word_index.items():
    train_embeddings_weights[index,:] = w2v[word] if word in w2v else np.random.rand(EMBEDDING_DIM)
print(train_embeddings_weights.shape)

# tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
#X_test = tokenizer.texts_to_sequences(X_test)

# print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
# print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
# # plt.hist([len(s) for s in X_train], bins=50)
# # plt.xlabel('length of samples')
# # plt.ylabel('number of samples')
# # plt.show()
# def below_threshold_len(max_len, nested_list):
#   cnt = 0
#   for s in nested_list:
#     if(len(s) <= max_len):
#         cnt = cnt + 1
#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
#
#
# max_len = 60
# below_threshold_len(max_len, X_train)
#
# X_train = pad_sequences(X_train, maxlen = max_len)
#X_test = pad_sequences(X_test, maxlen = max_len)

X_train = train_cnn_data
print(X_train)
y_train = train_data['label'].values

import re
from tensorflow.keras.layers import MaxPooling1D, Embedding, Dense, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
import tensorflow.keras.initializers as KI

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embeddings],
                                input_length=max_sequence_length, trainable=False)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs=[]
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
    # x = Dense(labels_index, activation='sigmoid')(l_merge)
    x = Dropout(0.5)(l_merge)
    x = Bidirectional(LSTM(100))(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model

model = ConvNet(train_embeddings_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, len(list(y_train)))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('snhs_cnnlstm1.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, epochs=30, callbacks=[es, mc], batch_size=10, validation_split=0.2)


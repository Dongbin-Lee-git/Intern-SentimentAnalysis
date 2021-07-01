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
    text = okt.morphs(keyword, stem=True)
    stopwords = ['블루핸즈','블루링크','블루', '핸즈','링크','도', '는', '다', '의', '가',
             '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네',
             '1','2','3','4','5','6','7','8','9','0','들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

    text = [word for word in text if not word in stopwords]
    return text

train_data['tokenized'] = np.load('X_save2.npy', allow_pickle=True)

#tqdm.pandas()
#train_data['tokenized'] = train_data['reviews'].progress_apply(lambda x: preprocword(x))
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

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
# X_test= test_data['tokenized'].values
# y_test = test_data['label'].values

#np.save('X_save2',X_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
#X_test = tokenizer.texts_to_sequences(X_test)

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
# plt.hist([len(s) for s in X_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))


max_len = 60
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len, truncating='post')
#X_test = pad_sequences(X_test, maxlen = max_len, truncating='post')

print(X_train.shape, y_train.shape)
print(X_train)

from tensorflow.keras.layers import Conv1D, Input, SpatialDropout1D, \
    Concatenate, GlobalMaxPooling1D, Flatten, Embedding,\
    BatchNormalization, Dense,GRU, LSTM, Bidirectional, Dropout, Activation,\
    MaxPooling1D, Conv2D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
#num_filters = 32

# 2stacked BiLSTM
# model = Sequential()
# model.add(Embedding(vocab_size, 128, input_length=max_len))
# model.add(Bidirectional(LSTM(64,dropout=0.3 ,return_sequences=True)))
# model.add(Bidirectional(LSTM(64, dropout=0.3)))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()
# conv_blocks = []
# for sz in [3, 4, 5]:
#     conv = Conv1D(filters = num_filters,
#                          kernel_size = sz,
#                          padding = "valid",
#                          activation = "relu",
#                          strides = 1)(model)
#     conv = GlobalMaxPooling1D()(conv)
#     conv = Flatten()(conv)
#     conv_blocks.append(conv)
# model.add(Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0])
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
#
#
# # -- Bi-LSTM
# # model = Sequential()
# # model.add(Embedding(vocab_size, 100))
# # model.add(Bidirectional(LSTM(128)))
# # model.add(Dense(1, activation='sigmoid'))
# # model.summary()


# # -- GRU
# # model = Sequential()
# # model.add(Embedding(vocab_size, 100))
# # #model.add(SpatialDropout1D(0.2))
# # model.add(Bidirectional(GRU(128)))
# # model.add(Dense(1, activation='sigmoid'))
# # model.summary()
#
#-----Bi-LSTM +  multi channel CNN
embedding_dim = 128
dropout_prob = (0.5, 0.8)
num_filters = 128
model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
z = Bidirectional(LSTM(128, return_sequences=True))(z)
z = Dropout(dropout_prob[0])(z)
conv_blocks = []
for sz in [3, 4, 5]:
    conv = Conv1D(filters = num_filters,
                         kernel_size = sz,
                         padding = "valid",
                         activation = "relu",
                         strides = 1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(128, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.summary()

#------BiLSTM + Attention Mechanism
# class BahdanauAttention(tf.keras.layers.Layer):
#   def __init__(self, units):
#     super(BahdanauAttention, self).__init__()
#     self.W1 = Dense(units)
#     self.W2 = Dense(units)
#     self.V = Dense(1)
#
#   def get_config(self):
#       config = super().get_config().copy()
#       config.update({
#           'W1': self.W1,
#           'W2': self.W2,
#           'V': self.V,
#       })
#       return config
#
#   def call(self, values, query): # 단, key와 value는 같음
#     # query shape == (batch_size, hidden size)
#     # hidden_with_time_axis shape == (batch_size, 1, hidden size)
#     # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
#     hidden_with_time_axis = tf.expand_dims(query, 1)
#
#     # score shape == (batch_size, max_length, 1)
#     # we get 1 at the last axis because we are applying score to self.V
#     # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#     score = self.V(tf.nn.tanh(
#         self.W1(values) + self.W2(hidden_with_time_axis)))
#
#     # attention_weights shape == (batch_size, max_length, 1)
#     attention_weights = tf.nn.softmax(score, axis=1)
#
#     # context_vector shape after sum == (batch_size, hidden_size)
#     context_vector = attention_weights * values
#     context_vector = tf.reduce_sum(context_vector, axis=1)
#
#     return context_vector, attention_weights
#
# sequence_input = Input(shape=(max_len,), dtype='int32')
# embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero = True)(sequence_input)
# lstm1 = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(embedded_sequences)
# lstm2, forward_h, forward_c, backward_h, backward_c = Bidirectional \
#   (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm1)
# print(lstm2.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)
# state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
# state_c = Concatenate()([forward_c, backward_c]) # 셀 상태
# attention = BahdanauAttention(64) # 가중치 크기 정의
# context_vector, attention_weights = attention(lstm2, state_h)
# dense1 = Dense(20, activation="relu")(context_vector)
# dropout = Dropout(0.5)(dense1)
# output = Dense(1, activation="sigmoid")(dropout)
# model = Model(inputs=sequence_input, outputs=output)
# model.summary()

# # -----multi channel CNN
# embedding_dim = 128
# dropout_prob = (0.5, 0.8)
# num_filters = 128
# model_input = Input(shape = (max_len,))
# z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
# z = Dropout(dropout_prob[0])(z)
# conv_blocks = []
#
# for sz in [3, 4, 5]:
#     conv = Conv1D(filters = num_filters,
#                          kernel_size = sz,
#                          padding = "valid",
#                          activation = "relu",
#                          strides = 1)(z)
#     conv = GlobalMaxPooling1D()(conv)
#     conv = Flatten()(conv)
#     conv_blocks.append(conv)
# z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
# z = Dropout(dropout_prob[1])(z)
# z = Dense(128, activation="relu")(z)
# model_output = Dense(1, activation="sigmoid")(z)
#
# model = Model(model_input, model_output)
# model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('snhs_rnn.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=64,callbacks=[es, mc],
                    validation_split=0.3, verbose=1, validation_steps=30)

# result = model.evaluate(X_test,y_test)

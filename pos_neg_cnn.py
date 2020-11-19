import re
import string
from konlpy.tag import Okt
import pandas as pd


data = pd.read_excel('naver_steam_hyundai.xlsx')
data.columns = ['Text', 'Label']
print(data.head())
print(data.Label.unique())
print(data.shape)
pos = []
neg = []
for i in data.Label:
    if i == 0:
        pos.append(0)
        neg.append(1)
    elif i==1:
        pos.append(1)
        neg.append(0)
    else:
        print(i)
print(len(pos))
print(len(neg))
data['Pos'] = pos
data['Neg'] = neg

print(data.head())


'''
Clean data
'''
#from eunjeon import Mecab

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
    text = mecab.morphs(keyword)
    stopwords = ['의', '가', '이', '은', '는', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    text = [word for word in text if not word in stopwords]
    print(text)
    return text

data['tokens'] = data['Text'].apply(lambda x: preprocword(x))
data['Text_Final'] = [' '.join(sen) for sen in data['tokens']]

data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]

# import matplotlib.pyplot as plt
# print('줄거리의 최대 길이 : {}'.format(max(len(l) for l in data)))
# print('줄거리의 평균 길이 : {}'.format(sum(map(len, data)) / len(data)))
# plt.hist([len(s) for s in data], bins=50)
# plt.xlabel('length of Data')
# plt.ylabel('number of Data')
# plt.show()

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
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data,test_size=0.1,random_state=42)
# data_train  = data
all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))

all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))

'''
Load glove to Word2Vec model
'''
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec('glove.6B.300d.txt', 'glove.6B.300d.txt.word2vec')

import numpy as np

from gensim import models
# word2vec_path = 'glove.6B.300d.txt.word2vec'
# word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
word2vec = models.KeyedVectors.load("word2vec")

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
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
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x,vectors,generate_missing=generate_missing))
    return list(embeddings)

'''
get embeddings
'''

training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

'''
Tokenize and Pad sequences
'''
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print("found %s unique tokens. " % len(train_word_index))

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embeddings_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word, index in train_word_index.items():
    train_embeddings_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embeddings_weights.shape)

test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

'''
Define CNN
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Embedding, Input, Conv1D, GlobalMaxPooling1D, concatenate, Dropout, Dense
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embeddings],
                                input_length=max_sequence_length, trainable=False)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequence = Dropout(0.5)(embedded_sequences)
    convs=[]
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequence)
        l_pool = GlobalMaxPooling1D()(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

   # x = Dropout(0.3)(l_merge)
    #x = Dense(128, activation='relu')(x)
    x = Dropout(0.8)(l_merge)
    x = Dense(128, activation='relu')(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input,preds)
    model.compile(loss = 'binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model

label_names = ['Pos', 'Neg']
y_train = data_train[label_names].values
x_train = train_cnn_data
y_tr = y_train

x_test = test_cnn_data
y_test = data_test[label_names].values

model = ConvNet(train_embeddings_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, len(list(label_names)))

'''
model checkpoint
'''
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model12_st.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
'''
train CNN
'''
num_epochs = 30
batch_size = 10
history = model.fit(x_train, y_tr, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[es,mc])


'''
model evaluate
'''
from tensorflow.keras.models import load_model
loaded_model = load_model('best_model12_st.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))


'''
save model
'''
# try:
#     model.save('Models/naver50000_hyundai_1step.h5', overwrite=True)
#     # model.save_weights('Models/1_layer_Weights_150_256.h5',overwrite=True)
# except:
#     print("Error in saving model.")
# print("Training complete...\n")

'''
hist plt
'''
# import matplotlib.pyplot as plt
# # 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # 학습 손실 값과 검증 손실 값을 플롯팅 합니다.
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# '''
# Test CNN
# '''
#
# predictions = model.predict(test_cnn_data, batch_size=10, verbose=1)
#
# labels = [1,0]
# predicton_labels=[]
# for p in predictions:
#     predicton_labels.append(labels[np.argmax(p)])
# print(sum(data_test.Label == predicton_labels)/len(predicton_labels))
#
# print(data_test.Label.value_counts())
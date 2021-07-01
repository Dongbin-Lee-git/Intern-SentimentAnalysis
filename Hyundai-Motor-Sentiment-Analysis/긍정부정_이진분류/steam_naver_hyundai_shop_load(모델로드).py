import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm
import numpy as np
vocab_size = 38351
max_len = 60
okt = Okt()



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
        keyword = keyword.replace(delword + delword, delword)
      else:
        break;
    return keyword

  def dltdot(keyword):
    while 1:
      if "…" in keyword:
        keyword = keyword.replace("…", "..")
      else:
        break;

    while 1:
      if "..." in keyword:
        keyword = keyword.replace("...", "..")
      else:
        break;
    return keyword

  keyword = text  # <-원문 넣을 곳
  keyword = delete(str(keyword))
  text = okt.morphs(keyword, stem=True)
  stopwords = ['블루핸즈', '블루링크', '블루', '핸즈', '링크', '도', '는', '다', '의', '가',
               '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네',
               '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

  text = [word for word in text if not word in stopwords]
  return text

# tqdm.pandas()
# train_data['tokenized'] = train_data['reviews'].progress_apply(lambda x: preprocword(x))
# X_train = train_data['tokenized'].values
#import h5py
loaded_model = load_model('snhs_rnn.h5')
# with h5py.File('snhs_rnn_dr02.h5', mode='r') as f:
#   # instantiate model
#   model_config = f.attrs.get('model_config')
#   print(model_config)


data_excel = pd.read_excel('201126_이노션샘플데이터.xlsx')
print(data_excel.head(5))
#data_excel.columns = ['Text']
#print(data_excel.head(5))
print(len(data_excel['reviews']))
x_save_load = np.load('X_save2.npy', allow_pickle=True)
tokenizer = Tokenizer(vocab_size, oov_token="OOV")
tokenizer.fit_on_texts(x_save_load)

f = open('result.txt', 'w', encoding='utf8')
f2 = open('result3.txt','w', encoding='utf8')
f.write('감성(test)\n')
f2.write('감성(test)\n')

def sentiment_predict(new_sentence):

  new_sentence = preprocword(new_sentence)
  #print(new_sentence)
  #print([new_sentence])
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  #print(encoded)
  pad_new = pad_sequences(encoded, maxlen = max_len, truncating='post') # 패딩
  #print(pad_new)
  score = float(loaded_model.predict(pad_new)) # 예측

  if(score > 0.5):
    f.write('긍정\n')
    if (score > 0.75):
      f2.write('긍정\n')
    else:
      f2.write('-\n')
  elif(score<0.5):
    f.write('부정\n')
    if(score < 0.25):
      f2.write('부정\n')
    else:
      f2.write('-\n')
  # print(new_sentence)


for idx, i in enumerate(data_excel['문장']):
  sentiment_predict(i)
  print(str(idx)+'/'+str(len(data_excel['문장'])))

f.close()
f2.close()

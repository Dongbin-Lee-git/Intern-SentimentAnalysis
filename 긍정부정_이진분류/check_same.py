import pandas as pd
import numpy as np

#df = pd.read_excel('200717_이노션샘플데이터_rdfScale.xlsx')
#df = pd.read_excel('현대자동차_긍부중_10월데이터.xlsx')
df = pd.read_excel('201126_이노션샘플데이터.xlsx')
print(df.head(5))
#df = df.iloc[1:]
df2 = pd.read_csv("result.txt", sep="\n", header=None)
df2.columns =['감성(test)']
print(df2.head(5))
#df2 = df2.iloc[1:]
cnt = 0
total = 0
for i, j in zip(df['감성'], df2['감성(test)']):
    if j != '-':
        total+=1
        if i==j:
            cnt += 1
print(cnt, total)
print((cnt/total) * 100)

df2 = pd.read_csv("result3.txt", sep="\n", header=None)
df2.columns =['감성(test)']
#df2=df2.iloc[1:]
print(df2.head(5))
cnt = 0
total = 0
for i, j in zip(df['감성'], df2['감성(test)']):
    if j != '-':
        total+=1
        if i==j:
            cnt += 1
print(cnt, total)
print((cnt/total) * 100)


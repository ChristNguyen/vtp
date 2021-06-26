#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import os

# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)


# In[2]:


# soi thử xem thằng tuổi thuê bao nó đang sai dữ liệu vào cỡ nào
df_cust = pd.read_csv('/u01/vtpay/truongnd26/20200616_test_1592296080479.csv',
                      dtype={'process_code': str, 'msisdn': str, 'ben_msisdn': str}, low_memory=False,
                      error_bad_lines=False)

df_cust = df_cust.drop(columns=['Unnamed: 10'])
df_cust

# In[3]:


# chuẩn chỉnh lại data
df_cust.loc[df_cust['gender'] == 'MALE', 'gender'] = '0'
df_cust.loc[df_cust['gender'] == 'FEMALE', 'gender'] = '1'
df_cust.loc[df_cust['gender'] == 'M', 'gender'] = '0'
df_cust.loc[df_cust['gender'] == 'F', 'gender'] = '1'


# bỏ bớt những thằng giới tính thứ 3 đi
# df_cust = df_cust[df_cust['gender'].isin(['0','1'])]


# In[30]:


# công thức tính tuổi của nó
def get_cust_age(d):
    x = d.split('-')
    try:
        return (2020 - int(x[0]))
    except ValueError:
        return None


# test
print(get_cust_age('1657-06-16 00:00:00'))
df_cust['age_year'] = df_cust['birthday'].astype(str).apply(get_cust_age)

# In[6]:


df_cust['age_year'].hist()

# In[7]:


x = pd.Series(range(-10, 100))
x = x.reset_index()
x.columns = ['num_customer', 'age_year']

for index, row in x.iterrows():
    row['num_customer'] = df_cust[df_cust['age_year'] == row['age_year']]['msisdn'].count()

x.plot(kind="bar", x='age_year', y=['num_customer'], figsize=(20, 8), grid=True)

# In[9]:


df_cust[df_cust['age_year'] <= 10]['msisdn'].nunique()  # 296069 dưới 10 tuổi

# df_cust[df_cust['age_year'] > 80 ]['msisdn'].nunique() #11275


# In[11]:


# nhìn riêng thằng gói 3
x = pd.Series(range(-10, 100))
x = x.reset_index()
x.columns = ['num_customer', 'age_year']

for index, row in x.iterrows():
    row['num_customer'] = df_cust[(df_cust['age_year'] == row['age_year']) & (df_cust['customer_type'] == 20)][
        'msisdn'].count()

x.plot(kind="bar", x='age_year', y=['num_customer'], figsize=(20, 8), grid=True)


# dưới 10 tuổi là 282432


# In[13]:


# df_cust[(df_cust['age_year'] <=10)  & (df_cust['customer_type'] == 20) ]['msisdn'].nunique() #37687 dưới 10 tuổi
# df_cust[(df_cust['age_year'] >80)  & (df_cust['customer_type'] == 20) ]['msisdn'].nunique() #7636 dưới 10 tuổi
# tính % có thể nhỏ nhưng tính về số lượng là to phết


# In[31]:


# thống kê theo ngày sinh trong năm
def get_cust_day_dob(d):
    x = d.split(' ')
    x = x[0].split('-')
    try:
        return (x[1] + '-' + x[2])
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_cust_day_dob('1657-06-16 00:00:00'))

# In[32]:


df_cust['day_of_birth'] = df_cust['birthday'].astype(str).apply(get_cust_day_dob)
df_cust

# In[29]:


df_cust[df_cust['customer_type'] == 20]['day_of_birth'].value_counts().plot.bar(stacked=True, figsize=(50, 3))
# chứng tỏ những thằng sinh ngày 1/1 nhiều đột biến gấp 10 lần thằng bình thường --> chứng tỏ fake


# In[7]:


# load data dữ liệu trong tháng 5
df_gd = pd.read_csv('/u01/vtpay/truongnd26/20200603_test_1591154700209.csv',
                    dtype={'process_code': str, 'msisdn': str, 'ben_msisdn': str}, low_memory=False,
                    error_bad_lines=False)
df_gd = df_gd.drop(columns=['Unnamed: 15'])
df_gd

# In[12]:


# xem thử bao nhiều thằng dưới 10 tuổi và trên
X = df_gd.merge(df_cust[~((df_cust['age_year'] > 14) & (df_cust['age_year'] < 90))], on='msisdn', how='inner',
                indicator=True)
X

# In[14]:


X[X['customer_type'] == 20]['msisdn'].nunique()
# vẫn còn 813007 GD tu nhung thue bao goi 3 ma tuoi từ 14 tuổi đổ xống hoặc trên 90 tuổi
# 68019 Khách hàng, rủi ro về mặt pháp lý --> chứng tỏ là sai thông tin rất nhiều --> rủi ro về mặt pháp lý


# In[18]:


# liệu có thể dùng model nào để dự đoán dữ liệu là sai không?
"""
Bóc các thông tin như
- ngày sinh, ví dụ sinh 1/1 xác suất tèo cao hơn
- họ, tên đệm, tên
- số dài của tên,số lượng từ của tên
- dùng model đoán thử xem nó có ra kg
"""
df_data = df_cust[df_cust['customer_type'] == 20]

# In[20]:


df_data = df_data.drop(columns=['msisdn', 'mobile_status', 'pin_status', 'end_date', 'customer_type'])

# In[32]:


df_data['day_of_birth'] = df_data['birthday'].astype(str).apply(get_cust_day_dob)
df_data


# In[35]:


# thống kê theo ngày sinh trong năm
def get_year(d):
    x = d.split('-')
    try:
        return int(x[0])
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_year('1657-06-16 00:00:00'))

# In[37]:


df_data['reg_age'] = df_data['issue_date'].astype(str).apply(get_year) - df_data['birthday'].astype(str).apply(get_year)
df_data

# In[40]:


df_data['len_name_char'] = df_data['cust_name'].astype(str).apply(len)
df_data


# In[42]:


def get_num_name_char_group(d):
    x = d.split(' ')
    try:
        return len(x)
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_num_name_char_group('Ho Anh Tien'))

# In[43]:


df_data['len_name_group'] = df_data['cust_name'].astype(str).apply(get_num_name_char_group)
df_data


# In[47]:


def get_first_name(d):
    x = d.split(' ')
    try:
        return x[0].lower()
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_first_name('Ho Anh Tien'))

# In[48]:


df_data['family_name'] = df_data['cust_name'].astype(str).apply(get_first_name)
df_data


# In[49]:


def get_last_name(d):
    x = d.split(' ')
    try:
        return x[len(x) - 1].lower()
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_last_name('Ho Anh Tien'))

# In[50]:


df_data['last_name'] = df_data['cust_name'].astype(str).apply(get_last_name)
df_data


# In[63]:


def get_mid_name(d):
    x = d.split(' ')
    if len(x) == 1:
        return d.lower()
    if len(x) == 2:
        return None

    tmp = x[1]
    for i in range(2, len(x) - 1):
        tmp = tmp + ' ' + x[i]
    try:
        return tmp.lower()
    except ValueError:
        return None
    except IndexError:
        return None


# test

print(get_mid_name('Ho Anh Tien'))

# In[64]:


df_data['mid_name'] = df_data['cust_name'].astype(str).apply(get_mid_name)
df_data

# In[67]:


# backup lại mai tính tiếp
df_data.to_csv(r'/u01/vtpay/truongnd26/cust_mobile_20200408.csv', index=False)

# In[78]:


# thử thằng tree xem sống kg
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

# In[81]:


feature_cols = ['day_of_birth', 'reg_age', 'age_year', 'len_name_char', 'len_name_group', 'family_name', 'last_name',
                'mid_name']
X = df_data[feature_cols]  # Features
y = df_data.gender  # Target variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# In[85]:


# muốn dùng tree vẫn phải hard code chuyển sang số
df_data['mid_name'].nunique()
# 80815 tên họ khác nhau, làm thành từ điển vẫn
# tên cũng 94646 tên khác nhau, siêu tổ hợp luôn
# tên đệm có đến 112551 nhát


# In[87]:


df_data.sample(50)

# In[93]:


df_data[df_data['len_name_group'] > 7].sample(50)

# In[92]:


df_data[df_data['len_name_group'] > 5]

# In[94]:


# nên chọn mẫu cỡ 10k để làm sạch dữ liệu trước
df_data.sample(10000).to_csv(r'df_data_name_10k_sample.csv')

# # Dự báo mô hình

# In[2]:


# df_data = pd.read_excel('df_data_name_10k_sample.xls', index_col=None, header=None)
# df_data.columns = ['mobile_id','issue_date','cust_name','gender','birthday','age_year','day_of_birth','reg_age','len_name_char','len_name_group','family_name','last_name','mid_name','fake']

df_data = pd.read_excel('15k_sample_name.xls', index_col=None, header=None)
df_data.columns = ['cust_name', 'fake']

# bỏ dòng tiều đề đi
df_data = df_data[df_data['cust_name'] != 'cust_name']
df_data

# In[6]:


# train thử model phát hiện fake xem sao
# input = df_data[['cust_name','gender','fake']]
# input.columns = ['name','gender','fake']
input = df_data[['cust_name', 'fake']]
input.columns = ['name', 'fake']
input['namelen'] = [len(str(i)) for i in input['name']]


def low(d):
    return d.lower()


print(low('S'))
input['name'] = input['name'].astype(str).apply(low)

# parameters
maxlen = 30
labels = 2
# lấy sẵn từ trước đỡ phải load lại
char_index = {'ù': 0, '~': 1, 'ｏ': 2, 'µ': 3, 'ả': 4, '}': 5, 'ａ': 6, '⁷': 7, ':': 8, 'ỵ': 9, '/': 10, 'ｕ': 11, 'ĭ': 12,
              'ĩ': 13, 'ƒ': 14, 'y': 15, '\x97': 16, 'ắ': 17, 'ë': 18, 'ǐ': 19, 'ầ': 20, '�': 21, 'j': 22, '_': 23,
              'è': 24, 'ü': 25, 'ｈ': 26, '=': 27, '½': 28, '\x98': 29, '\x92': 30, '̃': 31, 'ỡ': 32, '#': 33, ';': 34,
              'õ': 35, 'ẵ': 36, 'ø': 37, 'ữ': 38, 'q': 39, '\x95': 40, 'ỉ': 41, 'ộ': 42, '£': 43, 'ẳ': 44, '\x9d': 45,
              'α': 46, 'h': 47, 'ş': 48, 'ÿ': 49, '³': 50, 'ă': 51, '¤': 52, '̆': 53, '\x8c': 54, '0': 55, 'ї': 56,
              '9': 57, 'm': 58, 'ē': 59, '̀': 60, '¬': 61, ' ': 62, '<': 63, 'î': 64, 'ỹ': 65, 'ț': 66, '4': 67,
              '\x90': 68, ']': 69, 'l': 70, 'ŋ': 71, 'ứ': 72, 'ｉ': 73, '⁸': 74, '>': 75, 's': 76, '\x88': 77, '꧁': 78,
              '¹': 79, ',': 80, '\x8b': 81, 'ě': 82, '👉': 83, 'ề': 84, 'ỷ': 85, 'z': 86, '±': 87, 'ᴕ': 88, 'ị': 89,
              'ó': 90, 'r': 91, '`': 92, 'ê': 93, 'ｔ': 94, 'ố': 95, '̇': 96, 'ồ': 97, 'ậ': 98, 'b': 99, 'ễ': 100,
              'đ': 101, '6': 102, 'ä': 103, 'ｎ': 104, '¸': 105, 'ọ': 106, 'º': 107, 'ǒ': 108, '\x8d': 109, 'ą': 110,
              'ś': 111, 'ổ': 112, 'ỏ': 113, '\x80': 114, 'է': 115, '(': 116, 'þ': 117, 'ā': 118, '\\': 119, 'ö': 120,
              'ė': 121, '\x9e': 122, 'ｃ': 123, 'ɠ': 124, 'ẫ': 125, 'w': 126, '\u202d': 127, '°': 128, 'ợ': 129,
              'ï': 130, 'ỳ': 131, '%': 132, 'ũ': 133, '\x99': 134, 'á': 135, '7': 136, '\x93': 137, '[': 138, 'û': 139,
              '¯': 140, 'ƴ': 141, 'ẩ': 142, 'ń': 143, 'ự': 144, 'ử': 145, 'ｄ': 146, '￼': 147, '’': 148, '+': 149,
              '\x8e': 150, 'ì': 151, 'ů': 152, 'ĕ': 153, 'ų': 154, 'c': 155, 'ớ': 156, '"': 157, 'ｇ': 158, 'ｖ': 159,
              'ǔ': 160, 'é': 161, '*': 162, '\x86': 163, '@': 164, 'n': 165, '²': 166, 'u': 167, '¼': 168, 'ấ': 169,
              '\x9b': 170, 'ặ': 171, 'ň': 172, '2': 173, 'ı': 174, '-': 175, 'f': 176, 'ư': 177, '«': 178, '¥': 179,
              'x': 180, 'ệ': 181, '\u202c': 182, 'ạ': 183, 'd': 184, '»': 185, 'ủ': 186, 'ǹ': 187, '\x83': 188,
              '8': 189, 'ý': 190, 'ć': 191, 'ｌ': 192, 'ở': 193, '́': 194, '©': 195, '\t': 196, '̉': 197, 'i': 198,
              '¿': 199, '👍': 200, '😒': 201, '\x8a': 202, 'ụ': 203, '\x87': 204, '¶': 205, 'ể': 206, 'k': 207,
              'ẽ': 208, 'ῐ': 209, 'ú': 210, '\x85': 211, 'å': 212, 'ñ': 213, 'ẻ': 214, '®': 215, '´': 216, 'č': 217,
              'o': 218, 'ｒ': 219, 'ẹ': 220, '¨': 221, 'æ': 222, 't': 223, '\ufff3': 224, 'ð': 225, 'ŭ': 226, ')': 227,
              'ｍ': 228, 'ç': 229, 'ß': 230, '5': 231, '$': 232, '¦': 233, 'ū': 234, '꧂': 235, 'ò': 236, '\x89': 237,
              '\x82': 238, 'à': 239, 'END': 240, '§': 241, '¾': 242, 'ế': 243, '1': 244, 'ờ': 245, 'â': 246, '?': 247,
              'ŏ': 248, '&': 249, '\x91': 250, '\x9f': 251, '\x84': 252, 'ơ': 253, '3': 254, 'ª': 255, '\x81': 256,
              'g': 257, '·': 258, '̂': 259, '÷': 260, '\xad': 261, '×': 262, '¡': 263, '.': 264, 'ằ': 265, 'ő': 266,
              'í': 267, '\x9c': 268, 'ã': 269, 'ｙ': 270, '\x9a': 271, 'a': 272, "'": 273, 'v': 274, 'e': 275, 'ỗ': 276,
              'ô': 277, '\xa0': 278, 'ừ': 279, 'p': 280, '₫': 281, 'į': 282, '¢': 283, '\x94': 284, 'ｋ': 285, 'ℌ': 286,
              '̣': 287}
len_vocab = 288


def set_flag(i):
    tmp = np.zeros(len_vocab);
    tmp[i] = 1
    return (tmp)


# In[99]:


# xử lý vocal chung cho toàn tập để tránh thiếu
def low(d):
    return d.lower()


names = df_cust['cust_name'].astype(str).apply(low)
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)
print(len_vocab)

char_index = dict((c, i) for i, c in enumerate(vocab))
print(char_index)

# #input = X[X['fake'] == 0][['cust_name','gender','fake']]
# #input = pd.read_csv("gender_data.csv")
# #input.columns = ['name','m_or_f','race']
# input['namelen']= [len(str(i)) for i in input['name']]
# def low(d) :
#     return d.lower()
# print (low('S'))
# input['name'] = input['name'].astype(str).apply(low)
# input1 = input[(input['namelen'] >= 2) ]
# input

# In[101]:


# train test split
msk = np.random.rand(len(input)) < 0.8
train = input[msk]
test = input[~msk]

# take input upto max and truncate rest
# encode to vector space(one hot encoding)
# padd 'END' to shorter sequences
# also convert each index to one-hot encoding
train_X = []
train_Y = []


def set_flag(i):
    tmp = np.zeros(len_vocab);
    tmp[i] = 1
    return (tmp)


trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.fake:
    if i == 0:  # no fake
        train_Y.append([1, 0])
    else:
        train_Y.append([0, 1])

train_X = np.array(train_X)
np.asarray(train_X).shape
train_Y = np.array(train_Y)
np.asarray(train_X).shape

# In[102]:


test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_X.append(tmp)
for i in test.fake:
    if i == 0:
        test_Y.append([1, 0])
    else:
        test_Y.append([0, 1])
test_X = np.array(test_X)
print(np.asarray(test_X).shape)
test_Y = np.array(test_Y)
print(np.asarray(test_Y).shape)

# In[103]:


# build the model: 2 stacked LSTM
print('Build model...')
m_fake = Sequential()
m_fake.n_jobs = -1
m_fake.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len_vocab)))
m_fake.add(Dropout(0.2))
m_fake.add(LSTM(512, return_sequences=False))
m_fake.add(Dropout(0.2))
m_fake.add(Dense(2))
m_fake.add(Activation('softmax'))
m_fake.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1000
m_fake.fit(train_X, train_Y, batch_size=batch_size, nb_epoch=25, validation_data=(test_X, test_Y))

# In[104]:


import dill as pickle

# filename = '/u01/vtpay/truongnd26/lstm_name_fake_detection.sav'
# filename = '/u01/vtpay/truongnd26/lstm_name_fake_detection_v1.1.sav'
filename = '/u01/vtpay/truongnd26/lstm_name_fake_detection_v1.2.sav'
pickle.dump(m_fake, open(filename, 'wb'))

# In[ ]:


# áp data vào dự đoán xem sao
df_cust

# In[ ]:


# In[ ]:


# In[ ]:


# train để dự báo name và gender xem sao


# In[93]:


input1 = input[(input['namelen'] >= 2) & (input['fake'] == 0)]
input1.groupby('gender')['name'].count()

# In[39]:


# lưu ý kg set lại vocab cái này, đã dùng chung ở trên kia
names = input1['name']
gender = input1['gender']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)

# In[40]:


print(vocab)
print("vocab length is ", len_vocab)
print("length of input is ", len(input1))

# In[41]:


char_index = dict((c, i) for i, c in enumerate(vocab))
print(char_index)

# In[94]:


# train test split
msk = np.random.rand(len(input1)) < 0.8
train = input1[msk]
test = input1[~msk]

# In[71]:


np.asarray(train_X).shape

# In[72]:


train['namelen'].max()


# In[5]:


def set_flag(i):
    tmp = np.zeros(len_vocab);
    tmp[i] = 1
    return (tmp)


# In[44]:


set_flag(3)

# In[95]:


# take input upto max and truncate rest
# encode to vector space(one hot encoding)
# padd 'END' to shorter sequences
# also convert each index to one-hot encoding
train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.gender:
    if i == 0:  # male
        train_Y.append([1, 0])
    else:
        train_Y.append([0, 1])

train_X = np.array(train_X)
np.asarray(train_X).shape
train_Y = np.array(train_Y)
np.asarray(train_X).shape

# In[96]:


test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_X.append(tmp)
for i in test.gender:
    if i == 0:  # male
        test_Y.append([1, 0])
    else:
        test_Y.append([0, 1])
test_X = np.array(test_X)
print(np.asarray(test_X).shape)
test_Y = np.array(test_Y)
print(np.asarray(test_Y).shape)

# In[95]:


np.asarray(train_X).shape

# In[76]:


np.asarray(train_Y).shape

# In[97]:


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.n_jobs = -1
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len_vocab)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[81]:


np.asarray(test_X).shape
# print(np.asarray(test_Y).shape)


# In[98]:


batch_size = 1000
model.fit(train_X, train_Y, batch_size=batch_size, nb_epoch=25, validation_data=(test_X, test_Y))
# model.fit(test_X, test_Y,batch_size=batch_size,nb_epoch=10)
# model.fit(train_X, train_Y,batch_size=batch_size,nb_epoch=25)


# In[48]:


score, acc = model.evaluate(test_X, test_Y)
print('Test score:', score)
print('Test accuracy:', acc)

# In[127]:


import dill as pickle

filename = '/u01/vtpay/truongnd26/lstm_name_to_gender.sav'
pickle.dump(model, open(filename, 'wb'))

# In[100]:


# test thử xem dự đoán thế nào
# [1,0] male
# [0,1] female
name = ["nguyễn văn nam", "bùi thị thủy", "bùi sơn", "nguyễn thị hạnh"]
X = []
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred = model.predict(np.asarray(X))
pred


# In[127]:


def predict_gender(n):
    trunc_name = [i[0:maxlen] for i in n]
    X = []
    for i in trunc_name:

        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        X.append(tmp)

    pred = model.predict(np.asarray(X))

    if (pred[0][0] >= 0.7) & (pred[0][1] < 0.3):
        return 0  # male
    if (pred[0][0] < 0.3) & (pred[0][1] >= 0.7):
        return 1  # male

    return 9


name = ["nguyễn hùng lâm"]

print(predict_gender(name))


# In[11]:


def predict_fake(n):
    li = n.split(",#")
    trunc_name = [i[0:maxlen] for i in li]
    print(trunc_name)
    X = []
    for i in trunc_name:

        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        X.append(tmp)

    pred = m_fake.predict(np.asarray(X))
    # print(pred)
    if (pred[0][0] >= 0.7) & (pred[0][1] < 0.3):
        return 0  # not fake
    if (pred[0][0] < 0.3) & (pred[0][1] >= 0.7):
        return 1

    return 9


# name = "bankplus_841665833284"
name = "nguyễn văn bắc"

print(predict_fake(name))


# In[15]:


def predict_fake(n):
    x = []
    tmp = n[0:maxlen]
    tmp = [set_flag(char_index[j]) for j in str(tmp)]
    for k in range(0, maxlen - len(str(n))):
        tmp.append(set_flag(char_index["END"]))
    x.append(tmp)

    pred = m_fake.predict(np.asarray(x))
    # print(pred)
    if (pred[0][0] >= 0.7) & (pred[0][1] < 0.3):
        return 0  # not fake
    if (pred[0][0] < 0.3) & (pred[0][1] >= 0.7):
        return 1

    return 9


# name = "bankplus_841665833284"
name = "hà văn sở"

print(predict_fake(name))

# In[3]:


# load cấu hình từ mode đã save ra
import dill as pickle

filename = '/u01/vtpay/truongnd26/lstm_name_to_gender.sav'
model = pickle.load(open(filename, 'rb'))
model.n_jobs = -1

# In[177]:


import dill as pickle

filename = '/u01/vtpay/truongnd26/lstm_name_fake_detection.sav'
m_fake = pickle.load(open(filename, 'rb'))
m_fake.n_jobs = -1

# In[5]:


# parameters
maxlen = 30
labels = 2
# lấy sẵn từ trước đỡ phải load lại
char_index = {'ù': 0, '~': 1, 'ｏ': 2, 'µ': 3, 'ả': 4, '}': 5, 'ａ': 6, '⁷': 7, ':': 8, 'ỵ': 9, '/': 10, 'ｕ': 11, 'ĭ': 12,
              'ĩ': 13, 'ƒ': 14, 'y': 15, '\x97': 16, 'ắ': 17, 'ë': 18, 'ǐ': 19, 'ầ': 20, '�': 21, 'j': 22, '_': 23,
              'è': 24, 'ü': 25, 'ｈ': 26, '=': 27, '½': 28, '\x98': 29, '\x92': 30, '̃': 31, 'ỡ': 32, '#': 33, ';': 34,
              'õ': 35, 'ẵ': 36, 'ø': 37, 'ữ': 38, 'q': 39, '\x95': 40, 'ỉ': 41, 'ộ': 42, '£': 43, 'ẳ': 44, '\x9d': 45,
              'α': 46, 'h': 47, 'ş': 48, 'ÿ': 49, '³': 50, 'ă': 51, '¤': 52, '̆': 53, '\x8c': 54, '0': 55, 'ї': 56,
              '9': 57, 'm': 58, 'ē': 59, '̀': 60, '¬': 61, ' ': 62, '<': 63, 'î': 64, 'ỹ': 65, 'ț': 66, '4': 67,
              '\x90': 68, ']': 69, 'l': 70, 'ŋ': 71, 'ứ': 72, 'ｉ': 73, '⁸': 74, '>': 75, 's': 76, '\x88': 77, '꧁': 78,
              '¹': 79, ',': 80, '\x8b': 81, 'ě': 82, '👉': 83, 'ề': 84, 'ỷ': 85, 'z': 86, '±': 87, 'ᴕ': 88, 'ị': 89,
              'ó': 90, 'r': 91, '`': 92, 'ê': 93, 'ｔ': 94, 'ố': 95, '̇': 96, 'ồ': 97, 'ậ': 98, 'b': 99, 'ễ': 100,
              'đ': 101, '6': 102, 'ä': 103, 'ｎ': 104, '¸': 105, 'ọ': 106, 'º': 107, 'ǒ': 108, '\x8d': 109, 'ą': 110,
              'ś': 111, 'ổ': 112, 'ỏ': 113, '\x80': 114, 'է': 115, '(': 116, 'þ': 117, 'ā': 118, '\\': 119, 'ö': 120,
              'ė': 121, '\x9e': 122, 'ｃ': 123, 'ɠ': 124, 'ẫ': 125, 'w': 126, '\u202d': 127, '°': 128, 'ợ': 129,
              'ï': 130, 'ỳ': 131, '%': 132, 'ũ': 133, '\x99': 134, 'á': 135, '7': 136, '\x93': 137, '[': 138, 'û': 139,
              '¯': 140, 'ƴ': 141, 'ẩ': 142, 'ń': 143, 'ự': 144, 'ử': 145, 'ｄ': 146, '￼': 147, '’': 148, '+': 149,
              '\x8e': 150, 'ì': 151, 'ů': 152, 'ĕ': 153, 'ų': 154, 'c': 155, 'ớ': 156, '"': 157, 'ｇ': 158, 'ｖ': 159,
              'ǔ': 160, 'é': 161, '*': 162, '\x86': 163, '@': 164, 'n': 165, '²': 166, 'u': 167, '¼': 168, 'ấ': 169,
              '\x9b': 170, 'ặ': 171, 'ň': 172, '2': 173, 'ı': 174, '-': 175, 'f': 176, 'ư': 177, '«': 178, '¥': 179,
              'x': 180, 'ệ': 181, '\u202c': 182, 'ạ': 183, 'd': 184, '»': 185, 'ủ': 186, 'ǹ': 187, '\x83': 188,
              '8': 189, 'ý': 190, 'ć': 191, 'ｌ': 192, 'ở': 193, '́': 194, '©': 195, '\t': 196, '̉': 197, 'i': 198,
              '¿': 199, '👍': 200, '😒': 201, '\x8a': 202, 'ụ': 203, '\x87': 204, '¶': 205, 'ể': 206, 'k': 207,
              'ẽ': 208, 'ῐ': 209, 'ú': 210, '\x85': 211, 'å': 212, 'ñ': 213, 'ẻ': 214, '®': 215, '´': 216, 'č': 217,
              'o': 218, 'ｒ': 219, 'ẹ': 220, '¨': 221, 'æ': 222, 't': 223, '\ufff3': 224, 'ð': 225, 'ŭ': 226, ')': 227,
              'ｍ': 228, 'ç': 229, 'ß': 230, '5': 231, '$': 232, '¦': 233, 'ū': 234, '꧂': 235, 'ò': 236, '\x89': 237,
              '\x82': 238, 'à': 239, 'END': 240, '§': 241, '¾': 242, 'ế': 243, '1': 244, 'ờ': 245, 'â': 246, '?': 247,
              'ŏ': 248, '&': 249, '\x91': 250, '\x9f': 251, '\x84': 252, 'ơ': 253, '3': 254, 'ª': 255, '\x81': 256,
              'g': 257, '·': 258, '̂': 259, '÷': 260, '\xad': 261, '×': 262, '¡': 263, '.': 264, 'ằ': 265, 'ő': 266,
              'í': 267, '\x9c': 268, 'ã': 269, 'ｙ': 270, '\x9a': 271, 'a': 272, "'": 273, 'v': 274, 'e': 275, 'ỗ': 276,
              'ô': 277, '\xa0': 278, 'ừ': 279, 'p': 280, '₫': 281, 'į': 282, '¢': 283, '\x94': 284, 'ｋ': 285, 'ℌ': 286,
              '̣': 287}
len_vocab = 288


def set_flag(i):
    tmp = np.zeros(len_vocab);
    tmp[i] = 1
    return (tmp)


# In[9]:


X = df_cust[(df_cust['customer_type'] == 20) & (df_cust['mobile_status'] == 1)][
    ['cust_name', 'gender', 'mobile_id', 'issue_date']].reset_index()  # .sample(10000)


def low(d):
    return d.lower()


X['cust_name'] = X['cust_name'].astype(str).apply(low)
X

# In[63]:


# X = X.drop(columns = ['index'])
# X.to_csv(r'/u01/vtpay/truongnd26/customer_goi3_20200619.csv', index = False)
df_cust = pd.read_csv('/u01/vtpay/truongnd26/20200619_test_1592564940062.csv',
                      dtype={'process_code': str, 'msisdn': str, 'ben_msisdn': str, 'mobile_id': str}, low_memory=False,
                      error_bad_lines=False)
df_cust = df_cust.drop(columns=['Unnamed: 10'])


def low(d):
    return d.lower()


df_cust['cust_name'] = df_cust['cust_name'].astype(str).apply(low)

df_cust['pred_fake'] = np.nan
df_cust['pred_gender'] = np.nan
# df_cust['pred_fake'] = 10 #đánh dấu thằng nào chưa làm cho nó là 9

df_cust

# In[5]:


# backup lại cho chắc cái nào.
# df_cust.to_csv(r'/u01/vtpay/truongnd26/goi3_20200619_v1.1.csv', index = False)
# df_cust.to_csv(r'/u01/vtpay/truongnd26/goi3_20200619_v1.2.csv', index = False)
df_cust = pd.read_csv('/u01/vtpay/truongnd26/goi3_20200619_v1.2.csv',
                      dtype={'process_code': str, 'msisdn': str, 'ben_msisdn': str, 'mobile_id': str}, low_memory=False,
                      error_bad_lines=False)

# In[14]:


df_cust[(df_cust['pred_fake'] > 0.9) & (df_cust['mobile_status'] == 1)][
    ['msisdn', 'mobile_id', 'customer_type', 'cust_name', 'pred_fake']].to_csv(r'du_doan_fake.csv', index=False)
# df_cust[(df_cust['msisdn']==  '84974031315')   ]


# In[6]:


# xem thằng dự báo giới tính thì sao?
import datetime

before = datetime.datetime.now()
from_date = '2020-04-01'

j = 500
while j > 1:

    j = j - 1

    i = len(df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_gender'].isnull())])
    print(datetime.datetime.now())
    print(i)
    if i == 0:
        break

    if i > 5000:

        X = df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_gender'].isnull())][
            ['mobile_id', 'cust_name']].sample(5000).reset_index()
    else:
        X = df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_gender'].isnull())][
            ['mobile_id', 'cust_name']].reset_index()

    name = X['cust_name']
    pred = []

    trunc_name = [i[0:maxlen] for i in name]

    for i in trunc_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        pred.append(tmp)
    pred = model.predict(np.asarray(pred))

    pred = pd.DataFrame(pred)
    pred.columns = ['is_male', 'is_female']

    X['n_pred_gender'] = pred['is_male']

    df_cust = df_cust.merge(X[['mobile_id', 'n_pred_gender']], on='mobile_id', how='left', sort=True)

    df_cust.loc[(df_cust['pred_gender'].isnull()) & ~(df_cust['n_pred_gender'].isnull()), 'pred_gender'] = df_cust[
        'n_pred_gender']

    df_cust = df_cust.drop(columns=['n_pred_gender'])

after = datetime.datetime.now()
after - before

# In[105]:


df_cust['pred_fake'] = np.nan

# In[106]:


# dự báo chỗ thằng fake
import datetime

before = datetime.datetime.now()
from_date = '2020-04-01'

j = 500
while j > 1:

    j = j - 1

    i = len(df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_fake'].isnull())])
    print(datetime.datetime.now())
    print(i)
    if i == 0:
        break

    if i > 5000:

        X = df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_fake'].isnull())][
            ['mobile_id', 'cust_name']].sample(5000).reset_index()
    else:
        X = df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_fake'].isnull())][
            ['mobile_id', 'cust_name']].reset_index()
    name = X['cust_name']
    pred = []

    trunc_name = [i[0:maxlen] for i in name]

    for i in trunc_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        pred.append(tmp)
    pred = m_fake.predict(np.asarray(pred))

    pred = pd.DataFrame(pred)
    pred.columns = ['is_real', 'is_fake']

    X['n_pred_fake'] = pred['is_fake']

    df_cust = df_cust.merge(X[['mobile_id', 'n_pred_fake']], on='mobile_id', how='left', sort=True)

    df_cust.loc[(df_cust['pred_fake'].isnull()) & ~(df_cust['n_pred_fake'].isnull()), 'pred_fake'] = df_cust[
        'n_pred_fake']

    df_cust = df_cust.drop(columns=['n_pred_fake'])

after = datetime.datetime.now()
after - before

# In[14]:


# df_cust[df_cust['pred_fake'] > 0.9]#.value_counts()
df_cust[df_cust['pred_fake'] > 0.9].sample(20)
# df_cust
# nếu cắt từ 90% trở lên tỷ lệ chỉ khoản 1.2%, nhưng có đến 4385 TB
# nếu cắt từ 80% trở lên 1.4% và 5333 TB
# võ công cần làm rõ, vì làm giàu giữ liệu là ok


# In[118]:


# df_cust[df_cust['cust_name'] == 'lê lai']
df_cust[df_cust['msisdn'] == '84974031315']

# In[114]:


df_cust[df_cust['pred_fake'] > 0.9].sample(20)

# In[109]:


df_cust[(df_cust['pred_fake'] > 0.5) & (df_cust['pred_fake'] < 0.9)].sample(20)

# In[123]:


df_cust[df_cust['pred_fake'] > 0]['pred_fake'].hist()

# In[122]:


# len(df_cust[df_cust['pred_fake'] > 0.8] )
df_cust[df_cust['pred_fake'] > 0.1]['pred_fake'].hist()

# In[49]:


# xem thử độ tuổi nhóm fake và kg fake nó khác gì nhau không?
X = df_cust[(df_cust['issue_date'] > '2020-04-01') & (df_cust['pred_fake'] > 0.8)]
# nhìn riêng thằng gói 3
x = pd.Series(range(-10, 100))
x = x.reset_index()
x.columns = ['num_customer', 'age_year']

for index, row in x.iterrows():
    row['num_customer'] = X[(X['age_year'] == row['age_year'])]['mobile_id'].count()

x['num_cust_percent'] = x['num_customer'] / len(x)
x.plot(kind="bar", x='age_year', y=['num_cust_percent'], figsize=(20, 8), grid=True)

# x[x['age_year'] < 14]['num_cust_percent'].sum() #0.3090909090909091
# chỉ nhìn vào những thằng 0 tuổi thôi


# In[46]:


X = df_cust[(df_cust['issue_date'] > '2020-04-01') & (df_cust['pred_fake'] <= 0.5)]
# nhìn riêng thằng gói 3
x = pd.Series(range(-10, 100))
x = x.reset_index()
x.columns = ['num_customer', 'age_year']

for index, row in x.iterrows():
    row['num_customer'] = X[(X['age_year'] == row['age_year'])]['mobile_id'].count()

x['num_cust_percent'] = x['num_customer'] / len(x)
x.plot(kind="bar", x='age_year', y=['num_cust_percent'], figsize=(20, 8), grid=True)

# x[x['age_year'] < 14]['num_cust_percent'].sum() #6.454545454545455


# In[52]:


# xem thử ngày đăng ký thì sao?
X = df_cust[(df_cust['issue_date'] > '2020-04-01') & (df_cust['pred_fake'] > 0.8)]
X['day_of_birth'].value_counts().plot.bar(stacked=True, figsize=(50, 3))

# In[68]:


X = df_cust[(df_cust['issue_date'] > '2020-04-01') & (df_cust['pred_fake'] < 0.5)]
# len(X[X['day_of_birth'] == '01-01']) #1054 gấp 90 lần ngày bình thường ở nhóm fake, còn ở nhóm bình thường nó chỉ cao hơn cỡ 41 lần thôi
# len(X[X['day_of_birth'] != '01-01'])#['day_of_birth'].value_counts()
# len(X[X['day_of_birth'] != '01-01'])
35137 / (311664 / 365)

# In[53]:


# xem thử ngày đăng ký thì sao?
X = df_cust[(df_cust['issue_date'] > '2020-04-01') & (df_cust['pred_fake'] < 0.4)]
X['day_of_birth'].value_counts().plot.bar(stacked=True, figsize=(50, 3))

# In[26]:


df_cust['pred_fake'].hist()

# In[104]:


df_cust['pred_fake'].value_counts()
# df_cust
# tỷ lệ khai láo rơi vào cỡ 2%


# In[171]:


# df_cust[(df_cust['pred_gender'] <= 0.55) & (df_cust['gender'] == '0')] #6813
# df_cust[(df_cust['pred_gender'] > 0.55) & (df_cust['gender'] == '1')] #10027
# df_cust[(df_cust['pred_gender'] > 0.55) & (df_cust['gender'] == '0')] #112876
df_cust[(df_cust['pred_gender'] < 0.55) & (df_cust['gender'] == '1')]  # 101008


# In[135]:


def predict_gender(n):
    trunc_name = [i[0:maxlen] for i in n]
    X = []
    for i in trunc_name:

        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        X.append(tmp)

    pred = model.predict(np.asarray(X))

    print(pred)

    if (pred[0][0] >= 0.6) & (pred[0][1] < 0.4):
        return 0  # male
    if (pred[0][0] < 0.4) & (pred[0][1] >= 0.6):
        return 1  # male

    return 9


name = ["trịnh nguyễn tấn trung"]

print(predict_gender(name))

# In[179]:


# df_cust[ (df_cust['pred_fake'] == 10) ]
df_cust  # [ (df_cust['pred_fake'].isnull()) ]

# In[80]:


df_cust[(df_cust['issue_date'] > from_date) & (df_cust['pred_fake'] == 10)]

# In[15]:


df_cust[(df_cust['issue_date'] > '2020-06-10') & df_cust['pred_fake'].isnull()][['mobile_id', 'cust_name']].sample(
    5000).reset_index()

# In[38]:


df_cust[(df_cust['issue_date'] > '2020-06-10') & ~(df_cust['pred_fake'].isnull()) & df_cust['mobile_id'] == '40547398']

# In[33]:


X[X['mobile_id'] == '40547398']

# In[95]:


df_cust[df_cust['mobile_id'] == '36277142']

# In[24]:


X

# In[26]:


# X['mobile_id']#.nunique()
X['mobile_id'].value_counts()
# X = X.drop_duplicates(keep='last')
# 63193


# In[91]:


pred['n_pred_fake'] = np.nan
pred.loc[(pred['is_real'] >= 0.7) & (pred['is_fake'] < 0.3), 'n_pred_fake'] = 0
pred.loc[(pred['is_real'] < 0.3) & (pred['is_fake'] >= 0.7), 'n_pred_fake'] = 1
pred.loc[pred['n_pred_fake'].isnull(), 'n_pred_fake'] = 9
X['n_pred_fake'] = pred['n_pred_fake']
X

# In[16]:


df_cust

# In[1]:


df_cust

# In[ ]:


import tqdm
import concurrent.futures
import multiprocessing

num_processes = multiprocessing.cpu_count()

import datetime

before = datetime.datetime.now()

# Process the rows in chunks in parallel
with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
    # X['pred_fake'] = list(pool.map(predict_fake, X['cust_name'], chunksize=10)) # Without a progressbar
    X['pred_fake'] = list(
        tqdm.tqdm(pool.map(predict_fake, X['cust_name'], chunksize=10), total=X.shape[0]))  # With a progressb

after = datetime.datetime.now()
after - before

# In[1]:


X

# In[7]:


# tìm cách xuất nó ra file để test thử trên con colab nhat
msk = np.random.rand(len(input)) < 0.8
train = input[msk]
test = input[~msk]

test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_X.append(tmp)
for i in test.fake:
    if i == 0:
        test_Y.append([1, 0])
    else:
        test_Y.append([0, 1])
# test_X = np.array( test_X)
# print(np.asarray(test_X).shape)
# test_Y = np.array( test_Y)
# print(np.asarray(test_Y).shape)


# In[70]:


train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.fake:
    if i == 0:  # male
        train_Y.append([1, 0])
    else:
        train_Y.append([0, 1])

# In[8]:


np.savetxt('/u01/vtpay/truongnd26/test_y.txt', test_Y, fmt='%d')
# test_Y = np.loadtxt('/u01/vtpay/truongnd26/test1.txt', dtype=int)
# test_Y == test_YY


# In[9]:


np.asarray(test_X).shape

# In[46]:


f2d = np.array(test_X[0])
for i in range(1, len(test_X)):
    f2d = np.append(f2d, np.array(test_X[i]), axis=0)

# In[67]:


np.savetxt('test_x.txt', f2d, fmt='%d')
np.savetxt('test_y.txt', test_Y, fmt='%d')

# In[1]:


f2d = np.array(train_X[0])
for i in range(1, len(train_X)):
    f2d = np.append(f2d, np.array(train_X[i]), axis=0)

# In[ ]:


np.savetxt('train_x.txt', f2d, fmt='%d')
np.savetxt('train_y.txt', train_Y, fmt='%d')

# In[ ]:


np.savetxt('/u01/vtpay/truongnd26/test_x.txt', f2d, fmt='%d')

# In[52]:


f2d_ = np.loadtxt('/u01/vtpay/truongnd26/test_x.txt', dtype=int)
step = 30
test_Xb = []
for i in range(0, int(len(f2d_) / step)):
    tmp = f2d_[step * i:step * (i + 1)]
    test_Xb.append(tmp)

# In[59]:


# In[60]:


np.asarray(test_Xb).shape

# In[66]:


pd.DataFrame(test_Xb[0] == test_X[0])

# In[ ]:





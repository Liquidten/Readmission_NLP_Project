#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 06:17:49 2019

@author: sameepshah
"""
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords



# read positive data train
positive_dic = "/Archive/Readmission/Data/data_raw/train/yes/*.txt"
positive_files = glob.glob(positive_dic)
positive_examples = []
for file in positive_files:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        positive_examples.append(data)
positive_df = pd.DataFrame()
positive_df["notes"] = positive_examples
positive_df["label"] = 1

# read negative data train
negative_dic = "/Archive/Readmission/Data/data_raw/train/no/*.txt"
negative_files = glob.glob(negative_dic)
negative_examples = []
for file in negative_files:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        negative_examples.append(data)
negative_df = pd.DataFrame()
negative_df["notes"] = negative_examples
negative_df["label"] = -1

# concat
examples = pd.concat(objs=[positive_df, negative_df],
                     axis=0)

# clean train
examples["notes"] = examples["notes"].str.lower()
examples["notes"] = examples["notes"].str.strip()
examples["notes"] = examples["notes"].replace("[^\w\s]", "")
stop = stopwords.words('english')
examples["notes"] = examples["notes"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# variable assignment train
X_train = examples["notes"].values
y_train = examples["label"].values


# read positive data test
positive_examples_directory1 = "/Archive/Readmission/Data/data_raw/test/yes/*.txt"
positive_files1 = glob.glob(positive_examples_directory1)
positive_examples1 = []
for file in positive_files1:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        positive_examples1.append(data)
positive_df1 = pd.DataFrame()
positive_df1["notes"] = positive_examples1
positive_df1["label"] = 1

# read negative data test
negative_examples_directory1 = "/Archive/Readmission/Data/data_raw/test/no/*.txt"
negative_files1 = glob.glob(negative_examples_directory1)
negative_examples1 = []
for file in negative_files1:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        negative_examples1.append(data)
negative_df1 = pd.DataFrame()
negative_df1["notes"] = negative_examples1
negative_df1["label"] = 0

# concat test
examples1 = pd.concat(objs=[positive_df1, negative_df1],
                     axis=0)

# clean
examples1["notes"] = examples1["notes"].str.lower()
examples1["notes"] = examples1["notes"].str.strip()
examples1["notes"] = examples1["notes"].replace("[^\w\s]", "")
stop = stopwords.words('english')
examples1["notes"] = examples1["notes"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# variable assignment train
X_test = examples1["notes"].values
y_test = examples1["label"].values


# read positive data val
positive_examples_directory2 = "/Archive/Readmission/Data/data_raw/val/yes/*.txt"
positive_files2 = glob.glob(positive_examples_directory1)
positive_examples2 = []
for file in positive_files2:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        positive_examples2.append(data)
positive_df2 = pd.DataFrame()
positive_df2["notes"] = positive_examples2
positive_df2["label"] = 1

# read negative data val
negative_examples_directory2 = "/Archive/Readmission/Data/data_raw/val/no/*.txt"
negative_files2 = glob.glob(negative_examples_directory2)
negative_examples2 = []
for file in negative_files2:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        negative_examples2.append(data)
negative_df2 = pd.DataFrame()
negative_df2["notes"] = negative_examples2
negative_df2["label"] = 0

# concat test
examples2 = pd.concat(objs=[positive_df2, negative_df2],
                     axis=0)

# clean
examples2["notes"] = examples2["notes"].str.lower()
examples2["notes"] = examples2["notes"].str.strip()
examples2["notes"] = examples2["notes"].replace("[^\w\s]", "")
stop = stopwords.words('english')
examples2["notes"] = examples2["notes"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# variable assignment train
X_val = examples2["notes"].values
y_val = examples2["label"].values


# bag of words
tfidf = TfidfVectorizer(analyzer='word',
                        stop_words='english',
                        norm='l2',
                        use_idf=True,
                        smooth_idf=True,
                        lowercase=False)
                        #max_features = 400)
tfidf.fit(raw_documents=X_train)
X_train_matrix = tfidf.transform(raw_documents=X_train)
X_test_matrix = tfidf.transform(raw_documents=X_test)
X_val_matrix = tfidf.transform(raw_documents=X_val)



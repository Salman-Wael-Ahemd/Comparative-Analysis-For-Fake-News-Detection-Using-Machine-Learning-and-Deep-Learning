#!/usr/bin/env python
# coding: utf-8

# # Installing required libraries

# In[194]:


get_ipython().run_line_magic('load_ext', 'autotime')


# In[195]:


import warnings
from tqdm import TqdmWarning
warnings.filterwarnings('ignore', category=TqdmWarning)


# In[196]:


import numpy as np
import torch
import pandas as pd
import inflect
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from scipy.sparse import hstack, vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import time


# # Data Visualization and Initial Preprocessing for WELFake Dataset

# In[122]:


#Read Data
df = pd.read_csv("Dataset\WELFake_Dataset.csv")


# In[5]:


df.head(5)


# In[6]:


#Get column names
df.columns


# In[7]:


#Shape of Data
df.shape


# In[8]:


#Data info
df.info()


# In[9]:


# Drop ID Column
df = df.drop(df.columns[0], axis=1)
df.head(5)


# In[10]:


label_map = {0: "Fake", 1: "Real"}
df["label_name"] = df["label"].map(label_map)


# In[11]:


plt.figure(figsize=(7,5))
sns.countplot(data=df, x="label_name", hue="label_name", palette="viridis", legend=False)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[12]:


df["text_len"] = df["text"].apply(lambda x: len(str(x).split()))
df["title_len"] = df["title"].apply(lambda x: len(str(x).split()))


# In[13]:


classes = df["label"].unique()                     # [0, 1]
palette = sns.color_palette("husl", len(classes))  # distinct colors


# In[14]:


for i, c in enumerate(classes):
    class_name = label_map[c]
    subset = df[df["label"] == c]

    plt.figure(figsize=(10,6))
    sns.histplot(
        subset["text_len"],
        bins=40,
        kde=True,
        color=palette[i]
    )
    plt.title(f"Text Length Distribution for {class_name} News")
    plt.xlabel("Text Length (number of words)")
    plt.ylabel("Frequency")
    plt.show()


# In[15]:


for i, c in enumerate(classes):
    class_name = label_map[c]
    subset = df[df["label"] == c]

    plt.figure(figsize=(10,6))
    sns.histplot(
        subset["title_len"],
        bins=40,
        kde=True,
        color=palette[i]
    )
    plt.title(f"Title Length Distribution for {class_name} News")
    plt.xlabel("Title Length (number of words)")
    plt.ylabel("Frequency")
    plt.show()


# In[16]:


#Find NA Values
df.isna().sum()


# In[17]:


#Drop Na Text
df = df.dropna(subset=["text"])
df.isna().sum()


# In[18]:


#Fill Na Titles
df["title"] = df["title"].fillna("")
df.isna().sum()


# In[19]:


# Make title and case lower case
def text_lowercase(text):
	return text.lower()
df["title"] = df["title"].apply(text_lowercase)
df["text"] = df["text"].apply(text_lowercase)
df.head(5)


# In[20]:


#Remove white space
def remove_whitespace(text):
    return " ".join(text.split())
df["title"] = df["title"].apply(remove_whitespace)
df["text"] = df["text"].apply(remove_whitespace)
df.head(5)


# In[21]:


#Remove URLs
def remove_urls(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'http\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+|tinyurl\.com/\S+', '', text)
df["title"] = df["title"].apply(remove_urls)
df["text"] = df["text"].apply(remove_urls)
df.head(5)


# In[22]:


#Removal of HTML Tag
def remove_html(text):
    return re.sub(r'<[^>]+>', '', text)
df["title"] = df["title"].apply(remove_html)
df["text"] = df["text"].apply(remove_html)


# In[23]:


#Remove emojies
def remove_emoji(string):
    emoji_pattern = re.compile(
        "["                     
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # geometric shapes extended
        u"\U0001F800-\U0001F8FF"  # supplemental arrows
        u"\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
        u"\U0001FA00-\U0001FA6F"  # chess symbols, symbols & pictographs
        u"\U0001FA70-\U0001FAFF"  # symbols for games, puzzles
        u"\U00002700-\U000027BF"  # dingbats
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', string)
df["title"] = df["title"].apply(remove_emoji)
df["text"] = df["text"].apply(remove_emoji)
df.head(5)


# In[24]:


df["combined"] = df["title"] + " " + df["text"]
df.head(5)


# In[25]:


df_filtered = df[["combined","label"]]


# In[26]:


df_NN = df_filtered.copy()
df_ML = df_filtered.copy()


# In[27]:


df_ML.shape


# In[28]:


df_NN.shape


# # ISOT Dataset for Testing Models' Generalizability

# In[197]:


fake_df = pd.read_csv("Dataset_ISOT/Fake.csv")
true_df = pd.read_csv("Dataset_ISOT/True.csv")


# In[198]:


print(fake_df.shape)
print(true_df.shape)


# In[199]:


fake_df['label'] = 0
true_df['label'] = 1


# In[200]:


true_df


# In[201]:


df_exploring_isot = pd.concat([true_df, fake_df], ignore_index=True)


# In[202]:


df_exploring_isot = df_exploring_isot.reset_index(drop=True)


# In[203]:


df_exploring_isot.shape


# In[204]:


label_map = {0: "Fake", 1: "Real"}
df_exploring_isot["label_name"] = df_exploring_isot["label"].map(label_map)

plt.figure(figsize=(7, 5))
sns.countplot(data=df_exploring_isot, x="label_name", hue="label_name", palette="viridis", legend=False)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# In[205]:


df_exploring_isot["text_len"] = df_exploring_isot["text"].apply(lambda x: len(str(x).split()))
df_exploring_isot["title_len"] = df_exploring_isot["title"].apply(lambda x: len(str(x).split()))
classes = df_exploring_isot["label"].unique()                     # [0, 1]
palette = sns.color_palette("husl", len(classes))  # distinct colors
for i, c in enumerate(classes):
    class_name = label_map[c]
    subset = df_exploring_isot[df_exploring_isot["label"] == c]

    plt.figure(figsize=(10,6))
    sns.histplot(
        subset["text_len"],
        bins=40,
        kde=True,
        color=palette[i]
    )
    plt.title(f"Text Length Distribution for {class_name} News")
    plt.xlabel("Text Length (number of words)")
    plt.ylabel("Frequency")
    plt.show()


# In[206]:


for i, c in enumerate(classes):
    class_name = label_map[c]
    subset = df_exploring_isot[df_exploring_isot["label"] == c]

    plt.figure(figsize=(10,6))
    sns.histplot(
        subset["title_len"],
        bins=40,
        kde=True,
        color=palette[i]
    )
    plt.title(f"Title Length Distribution for {class_name} News")
    plt.xlabel("Title Length (number of words)")
    plt.ylabel("Frequency")
    plt.show()


# In[207]:


isot_test_df = pd.concat([fake_df, true_df], ignore_index=True)
print(isot_test_df["label"].value_counts())


# In[208]:


isot_test_df.shape


# In[209]:


isot_test_df = isot_test_df.drop(columns=['subject', 'date'])


# In[210]:


isot_test_df.isnull().sum()


# In[211]:


# Make title and case lower case
def text_lowercase(text):
	return text.lower()
isot_test_df["title"] = isot_test_df["title"].apply(text_lowercase)
isot_test_df["text"] = isot_test_df["text"].apply(text_lowercase)
isot_test_df.head(5)


# In[212]:


#Remove white space
def remove_whitespace(text):
    return " ".join(text.split())
isot_test_df["title"] = isot_test_df["title"].apply(remove_whitespace)
isot_test_df["text"] = isot_test_df["text"].apply(remove_whitespace)
isot_test_df.head(5)


# In[213]:


#Remove URLs
def remove_urls(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'http\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+|tinyurl\.com/\S+', '', text)
isot_test_df["title"] = isot_test_df["title"].apply(remove_urls)
isot_test_df["text"] = isot_test_df["text"].apply(remove_urls)
isot_test_df.head(5)


# In[214]:


#Removal of HTML Tag
def remove_html(text):
    return re.sub(r'<[^>]+>', '', text)
isot_test_df["title"] = isot_test_df["title"].apply(remove_html)
isot_test_df["text"] = isot_test_df["text"].apply(remove_html)


# In[215]:


#Remove emojies
def remove_emoji(string):
    emoji_pattern = re.compile(
        "["                     
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # geometric shapes extended
        u"\U0001F800-\U0001F8FF"  # supplemental arrows
        u"\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
        u"\U0001FA00-\U0001FA6F"  # chess symbols, symbols & pictographs
        u"\U0001FA70-\U0001FAFF"  # symbols for games, puzzles
        u"\U00002700-\U000027BF"  # dingbats
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', string)
isot_test_df["title"] = isot_test_df["title"].apply(remove_emoji)
isot_test_df["text"] = isot_test_df["text"].apply(remove_emoji)
isot_test_df.head(5)


# In[216]:


isot_test_df["combined"] = isot_test_df["title"] + " " + isot_test_df["text"]
isot_test_df.head(5)


# In[217]:


df_filtered_isot = isot_test_df[["combined","label"]]


# In[219]:


df_NN_isot = df_filtered_isot.copy()
df_ML_isot = df_filtered_isot.copy()


# # ML model

# ## WELFake Dataset further Preprocessing

# In[51]:


# Further Text Preprocessing for DF used for ML model
def remove_punctuation(text):
    allowed = "-'" 
    punct = ''.join([c for c in string.punctuation if c not in allowed])
    return text.translate(str.maketrans('', '', punct))
df_ML["combined"] = df_ML["combined"].apply(remove_punctuation)


# In[52]:


df_ML


# In[53]:


nltk.data.path.append("D:/NLTK")
nltk.data.path = ["D:/NLTK"]
print(nltk.data.path)


# In[54]:


#Remove StopWords
stop_words = set(stopwords.words("english"))
def remove_stopwords(text):
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)
df_ML["combined"] = df_ML["combined"].apply(remove_stopwords)


# In[55]:


df_ML.head(5)


# In[56]:


#Extract POS Tags
def extract_pos_features(text):
    tokens = text.split()
    tags = pos_tag(tokens)

    counts = {
        "noun_count": 0,
        "verb_count": 0,
        "adj_count": 0,
        "adv_count": 0,
        "modal_count": 0,
    }

    for _, t in tags:
        if t.startswith("N"): counts["noun_count"] += 1
        elif t.startswith("V"): counts["verb_count"] += 1
        elif t.startswith("J"): counts["adj_count"] += 1
        elif t.startswith("R"): counts["adv_count"] += 1
        elif t == "MD": counts["modal_count"] += 1

    return counts


# In[57]:


pos_feats = df_ML["combined"].apply(extract_pos_features)
pos_df = pos_feats.apply(pd.Series)


# In[58]:


pos_feats


# In[59]:


pos_df


# In[60]:


df_ML = pd.concat([df_ML, pos_df], axis=1)


# In[61]:


df_ML.head(5)


# In[62]:


#Lemmatization based on POS
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN
        
def lemmatize_text(text):
    tokens = text.split()
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
    ]
    return " ".join(lemmas)
df_ML["combined"] = df_ML["combined"].apply(lemmatize_text)


# In[64]:


df_ML.head(5)


# ### Downloading The Preprocessed WELFake Dataset for faster testing

# In[ ]:


# Save df_ML data to not run code again
df_ML.to_pickle("df_ML_preprocessed.pkl")


# In[126]:


df_ML = pd.read_pickle("df_ML_preprocessed.pkl")


# In[127]:


#Split Data into Training and Testing
X_text = df_ML["combined"].values
X_pos  = df_ML[["noun_count","verb_count","adj_count","adv_count","modal_count"]].values
y      = df_ML["label"].values


# ## ISOT Dataset further Preprocessing

# In[221]:


# Further Text Preprocessing 
def remove_punctuation(text):
    allowed = "-'" 
    punct = ''.join([c for c in string.punctuation if c not in allowed])
    return text.translate(str.maketrans('', '', punct))
df_ML_isot["combined"] = df_ML_isot["combined"].apply(remove_punctuation)


# In[222]:


df_ML_isot.tail()


# In[223]:


#Remove StopWords
stop_words = set(stopwords.words("english"))
def remove_stopwords(text):
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)
df_ML_isot["combined"] = df_ML_isot["combined"].apply(remove_stopwords)


# In[224]:


df_ML_isot.tail()


# In[225]:


#Extract POS Tags
def extract_pos_features(text):
    tokens = text.split()
    tags = pos_tag(tokens)

    counts = {
        "noun_count": 0,
        "verb_count": 0,
        "adj_count": 0,
        "adv_count": 0,
        "modal_count": 0,
    }

    for _, t in tags:
        if t.startswith("N"): counts["noun_count"] += 1
        elif t.startswith("V"): counts["verb_count"] += 1
        elif t.startswith("J"): counts["adj_count"] += 1
        elif t.startswith("R"): counts["adv_count"] += 1
        elif t == "MD": counts["modal_count"] += 1

    return counts


# In[227]:


pos_feats_isot = df_ML_isot["combined"].apply(extract_pos_features)
pos_df_isot = pos_feats_isot.apply(pd.Series)


# In[228]:


pos_feats_isot


# In[229]:


pos_df_isot


# In[230]:


df_ML_isot = pd.concat([df_ML_isot, pos_df_isot], axis=1)


# In[231]:


df_ML_isot.tail(5)


# In[232]:


#Lemmatization based on POS
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN
        
def lemmatize_text(text):
    tokens = text.split()
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
    ]
    return " ".join(lemmas)
df_ML_isot["combined"] = df_ML_isot["combined"].apply(lemmatize_text)


# In[233]:


df_ML_isot.head(5)


# In[234]:


df_ML_isot.tail(5)


# ### Downloading The Preprocessed IOST Dataset for faster testing

# In[235]:


# Save df_ML data to not run code again
df_ML_isot.to_pickle("df_ML_isot_preprocessed.pkl")


# In[128]:


df_ML_isot = pd.read_pickle("df_ML_isot_preprocessed.pkl")


# In[236]:


#Split Data into Training and Testing
X_text_isot = df_ML_isot["combined"].values
X_pos_isot  = df_ML_isot[["noun_count","verb_count","adj_count","adv_count","modal_count"]].values
y_isot      = df_ML_isot["label"].values


# ## Training SVC Model on WELFake Dataset

# In[243]:


#Training set and test set that will be split for validation and test set
X_text_train, X_text_temp, X_pos_train, X_pos_temp, y_train, y_temp = train_test_split(
    X_text,
    X_pos,
    y,
    train_size=0.8,
    random_state=2025,
    stratify=y
)


# In[244]:


# Getting test and validation sets
# The splits are 80% Training, 10% Validation, 10% Testing
X_text_val, X_text_test, X_pos_val, X_pos_test, y_val, y_test = train_test_split(
    X_text_temp,
    X_pos_temp,
    y_temp,
    test_size=0.5,
    random_state=2025,
    stratify=y_temp
)


# In[245]:


#Setting up the TD-IDF Object
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)


# In[246]:


#Applying TF-IDF to Text Data
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_val   = tfidf.transform(X_text_val)
X_tfidf_test  = tfidf.transform(X_text_test)


# In[247]:


# Make POS Statstics we got into Sparse matrix to stack them and use them for the model
X_pos_train_sparse = sparse.csr_matrix(X_pos_train)
X_pos_val_sparse   = sparse.csr_matrix(X_pos_val)
X_pos_test_sparse  = sparse.csr_matrix(X_pos_test)

X_train_final = hstack([X_tfidf_train, X_pos_train_sparse])
X_val_final   = hstack([X_tfidf_val,   X_pos_val_sparse])
X_test_final  = hstack([X_tfidf_test,  X_pos_test_sparse])


# In[248]:


X_train_final.shape


# 50000 features from TF-IDF and 5 Features from POS

# In[249]:


#Import SVC model and fit
svc = LinearSVC(C=2.0)
svc.fit(X_train_final, y_train)


# In[255]:


# Predict on validation
y_val_pred = svc.predict(X_val_final)

print("====== Linear SVM (Validation) ======")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))


# In[256]:


# Predict on test
y_test_pred = svc.predict(X_test_final)

print("====== Linear SVM (Test) ======")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


# In[257]:


cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM Model")
plt.show()


# In[258]:


# Try to add validation and training and train on both then see performance on test like:
X_train_final_2 = vstack([X_train_final, X_val_final])
y_train_final_2 = np.concat([y_train, y_val])


# In[259]:


#Import SVC model and fit
svc = LinearSVC(C=2.0)
svc.fit(X_train_final_2, y_train_final_2)


# In[260]:


# Predict on test
y_test_pred = svc.predict(X_test_final)

print("====== Linear SVM (Test) ======")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


# In[261]:


cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVM Model")
plt.show()


# ## Evaluating SVC Model on ISOT Dataset

# In[263]:


X_tfidf_test_isot = tfidf.transform(X_text_isot)


# In[269]:


# Make POS Statstics we got into Sparse matrix to stack them and use them for the model
X_pos_test_sparse_isot  = sparse.csr_matrix(X_pos_isot)

X_test_final_isot  = hstack([X_tfidf_test_isot,  X_pos_test_sparse_isot])


# In[270]:


X_test_final_isot.shape


# In[271]:


# Predict on test
y_test_pred_isot = svc.predict(X_test_final_isot)

print("====== Linear SVM (Test) ======")
print("Test Accuracy:", accuracy_score(y_isot, y_test_pred_isot))
print("\nClassification Report:\n", classification_report(y_isot, y_test_pred_isot))
print("\nConfusion Matrix:\n", confusion_matrix(y_isot, y_test_pred_isot))


# # Checking for GPU Availability

# In[178]:


#Check Pytorch version
import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# # DL Model

# ## Setting up Model and Tokenizer

# In[170]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)


# ### Saved Model Parameters and Tokenizer after training for immediate testing

# In[171]:


SAVE_DIR = "distilbert_fake_news"

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)

model.to(device)


# ### If you want to download tokenizer and model agian for training

# In[192]:


# Select a model
MODEL_NAME = "distilbert-base-uncased"


# In[193]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[194]:


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)


# ## Functions for training and evaluating the model

# In[172]:


MAX_LEN = 512
BATCH_SIZE = 32

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# In[173]:


EPOCHS = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# In[174]:


def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(data_loader, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Step {step}/{len(data_loader)} | "
                f"Loss: {loss.item():.4f}")

    avg_loss = total_loss/len(data_loader)
    return avg_loss


# In[175]:


def eval_model(model, data_loader, device, return_cm=False):
    model.eval()
    preds = []
    true_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim = 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', pos_label= 1)

    if return_cm:
        cm = confusion_matrix(true_labels, preds)
        return avg_loss, acc, precision, recall, f1, cm, true_labels, preds

    return avg_loss, acc, precision, recall, f1


# ## Training and Testing on WELFake Dataset

# In[176]:


texts = df_NN['combined'].astype(str).values
labels = df_NN['label'].values

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, train_size=0.8, stratify=labels, random_state=2025)
X_val , X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state= 2025)

print("Train size : " , len(X_train))
print("Validation size : " , len(X_val))
print("Test size : " , len(X_test))


# In[177]:


train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(X_val, y_val, tokenizer, MAX_LEN)
test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle=True)


# In[178]:


total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# In[203]:


best_val_f1 = 0.0
best_state_dict = None

for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS} ======')

    start_time = time.time()
    
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = eval_model(model, val_loader, device)

    epoch_time = time.time() - start_time
    
    print(f"model validation loss is {val_loss:.4f}")
    print(f"model accuracy is {val_acc:.4f}")
    print(f"model precision loss is {val_prec:.4f}")
    print(f"model recall loss is {val_rec:.4f}")
    print(f"model f1-score loss is {val_f1:.4f}")
    print(f"Epoch time: {epoch_time/60:.2f} minutes")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_dict = model.state_dict()
        print("New best model has been saved using F1-Score")


# ## If you just trained the model

# In[204]:


if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    
test_loss, test_acc, test_prec, test_rec, test_f1, cm, y_true, y_pred = eval_model(model, test_loader, device, return_cm=True)
print("\n===== FINAL TEST RESULTS (DistilBERT) =====")
print(f"Test loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}")
print(f"Test Prec: {test_prec:.4f}")
print(f"Test Rec:  {test_rec:.4f}")
print(f"Test F1:   {test_f1:.4f}")


# ## If you reloaded the parameters and want fast testing

# In[179]:


test_loss, test_acc, test_prec, test_rec, test_f1, cm, y_true, y_pred = eval_model(model, test_loader, device, return_cm=True)
print("\n===== FINAL TEST RESULTS (DistilBERT) =====")
print(f"Test loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}")
print(f"Test Prec: {test_prec:.4f}")
print(f"Test Rec:  {test_rec:.4f}")
print(f"Test F1:   {test_f1:.4f}")


# ## Evaluating performance of the model

# In[180]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – DistilBERT (Test Set)")
plt.show()


# ## How to save and reload model's parameters and tokenizer for fast testing

# In[227]:


'''# Save the parameters needed to reload them later instead of training
SAVE_DIR = "distilbert_fake_news"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Reload the parameters
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)

model.to(device)
'''


# ## ISOT Dataset

# In[181]:


X_isot = df_NN_isot['combined'].astype(str).values
y_isot = df_NN_isot['label'].values


# In[182]:


isot_dataset = FakeNewsDataset(texts=X_isot, labels= y_isot, tokenizer=tokenizer, max_len=MAX_LEN)
isot_loader = DataLoader(isot_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[183]:


isot_loss, isot_acc, isot_prec, isot_rec, isot_f1, isot_cm, y_true, y_pred = eval_model(model,isot_loader,device,return_cm=True)


# In[184]:


print("\n===== ISOT DATASET (GENERALIZATION TEST) =====")
print(f"Loss:      {isot_loss:.4f}")
print(f"Accuracy:  {isot_acc:.4f}")
print(f"Precision: {isot_prec:.4f}")
print(f"Recall:    {isot_rec:.4f}")
print(f"F1-score:  {isot_f1:.4f}")


# In[192]:


disp = ConfusionMatrixDisplay(
    confusion_matrix=isot_cm,
    display_labels=["Fake", "Real"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – ISOT Dataset (Generalization)")
plt.show()


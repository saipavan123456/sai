import tkinter as tk
from tkinter import messagebox
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))
snow = SnowballStemmer('english')

def preprocess_text(message):
    message = message.lower().strip()
    cleaner = re.compile('<.*?>')
    message = re.sub(cleaner, '', message)
    message = re.sub(r'[^a-zA-Z0-9]', ' ', message)
    message = re.sub(r'\d+', '', message)
    words = [snow.stem(word) for word in message.split() if word not in stop_words]
    return ' '.join(words)

def check_spam(user_input):
    # Read the dataset
    df = pd.read_csv(r'C:\Users\pavan\Desktop\spam.csv', encoding="ISO-8859-1")

    # Selecting relevant columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Text preprocessing
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Model training using CountVectorizer and Logistic Regression
    x = df['processed_message'].values
    y = df['label'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(max_features=5000)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train_vectorized, y_train)

    processed_message = preprocess_text(user_input)
    message_vectorized = vectorizer.transform([processed_message])
    prediction = model.predict(message_vectorized)

    if prediction[0] == 'ham':
        result = "Not Spam"
    else:
        result = "Spam"

    messagebox.showinfo("Prediction", f"This message is {result}")

# Tkinter GUI\
def on_enter(event=None):
    user_input = entry.get()
    check_spam(user_input)

root = tk.Tk()
root.title("Spam Detector")

label = tk.Label(root, text="Enter a message:")
label.pack()

entry = tk.Entry(root)
entry.pack()
entry.bind("<Return>", on_enter)

button = tk.Button(root, text="Check", command=on_enter)
button.pack()

root.mainloop()
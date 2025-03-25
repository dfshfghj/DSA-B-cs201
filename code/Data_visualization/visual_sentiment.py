import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
from projection import orthogonalization, projection


def preprocess_text(text):
    import re
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'@.*? ', ' ', text)
    text = re.sub(r'#.*? ', ' ', text)
    text = re.sub(r'&.*?;', ' ', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'http\S+|www\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text=text.lower()
    words=text.split()
    return ' '.join(words)

dataset=pd.read_csv("Sentiment_Analysis_Dataset.csv", on_bad_lines='skip')
dataset['SentimentText']=dataset['SentimentText'].apply(preprocess_text)
label=dataset['Sentiment']
data=dataset['SentimentText']
xtrain_bow, xtest_bow, ytrain, ytest = train_test_split(data, label, random_state=42, test_size=0.2)
vectorizer = TfidfVectorizer(max_df=0.8)
xtrain_vec = vectorizer.fit_transform(xtrain_bow)
xtest_vec = vectorizer.transform(xtest_bow)
'''
model = SVC(kernel='linear', C=1.0)
model.fit(xtrain_vec, ytrain)
ypred = model.predict(xtest_vec)
print("Accuracy:", accuracy_score(ytest, ypred))
print("\nClassification Report:\n", classification_report(ytest, ypred))
joblib.dump(model, 'sentiment_model.sav')
joblib.dump(vectorizer, 'sentiment_vectorizer.sav')'
'''
#print(model.coef_)
#print(model.intercept_)
model = joblib.load('sentiment_model.sav')
normal = model.coef_.toarray()[0]
normal = normal / np.linalg.norm(normal)
alpha = orthogonalization(normal, 3)
X = xtest_vec.toarray()
c = np.array(projection(alpha, X))
c = c.T
x, y, z = c
x_a, y_a, z_a = [], [], []
x_b, y_b, z_b = [], [], []
ytest = ytest.tolist()
for i in range(len(x)):
    if ytest[i] == 0:
        x_a.append(x[i])
        y_a.append(y[i])
        z_a.append(z[i])
    else:
        x_b.append(x[i])
        y_b.append(y[i])
        z_b.append(z[i])
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(x_a, y_a, z_a)
ax.scatter(x_b, y_b, z_b)
plt.show()
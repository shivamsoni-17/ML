import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import streamlit as st


df = pd.read_csv('credit_customers (1).csv')
st.header("_Credit_Risk_   :red[Prediction Model]")
st.subheader(':blue[Customer_Credit_Data: ]')
st.dataframe(df.head())

nav = st.sidebar.radio("Select Countplot Feature",["credit_history","savings_status","housing","existing_credits"])


st.subheader(':blue[Based on] '+ nav)
f1 = plt.figure(figsize=(3,3))
sns.countplot(x='class',hue=nav,data=df)
st.pyplot(f1)

# Handling null values
df = df.dropna()

# Handling duplicate records
df = df.drop_duplicates()


# Label Encoding
categorical_cols = ['checking_status', 'credit_history', 'purpose','credit_amount','savings_status','employment', 'existing_credits', 'housing', 'job', 'own_telephone','class']
df1 = df[['checking_status', 'credit_history', 'purpose','credit_amount','savings_status','employment', 'existing_credits', 'housing', 'job', 'own_telephone','class']].copy()

lb = LabelEncoder()

for col in categorical_cols:
    df1[col] = lb.fit_transform(df[col])

st.subheader('\n :blue[Data after LabelEncoding: ]')
st.dataframe(df1.head(5))



x = df1.drop('class', axis=1)
y = df1['class']

st.write('Shape of features:\n')
st.write(x.shape,y.shape)


# Split data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


classifier_name = st.sidebar.selectbox('Choose classifier for Prediction',('KNN', 'SVM','Logistic Regression'))



# Parameter Selection
def select_param(clf_name):
    params = {}
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        ker = st.sidebar.selectbox('Select kernel',('linear','rbf'))
        params['kernel'] = ker
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        None
    return params

params = select_param(classifier_name)

# Classifier Selection
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(kernel=params['kernel'],C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = LogisticRegression(max_iter=10000)
    return clf

model = get_classifier(classifier_name, params)
model.fit(x_train,y_train)


y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)




st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write(cm)
st.write(f'Classification Report\n',classification_report(y_test,y_pred))
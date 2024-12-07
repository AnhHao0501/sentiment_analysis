import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


data =pd.read_csv('Danh_gia_cleaned_updated.csv')
data['content_new']=data['content_new'].fillna('')
data['sentiment'] = np.where(data['so_sao']<=3, 1, 0)
X = data['content_new']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
count = TfidfVectorizer(max_features=5000) 
X_train = count.fit_transform(X_train)
X_test = count.transform(X_test)

svc_model = SVC(C=10, gamma='scale', kernel='rbf', probability=True)
svc_model.fit(X_train,y_train)
y_predict = svc_model.predict(X_test)

score_train = svc_model.score(X_train,y_train)
score_test = svc_model.score(X_test,y_test)
acc = accuracy_score(y_test,y_predict)
cm = confusion_matrix(y_test, y_predict, labels=[0, 1])

cr = classification_report(y_test, y_predict)

y_prob = svc_model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

pkl_filename = "svc.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(svc_model, file)

pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)

with open(pkl_filename, 'rb') as file:  
    svc = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

st.title("Data Science Project")
st.write("## Ham vs Spam")

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Nguyễn Văn A & Trần Thị B""")
st.sidebar.write("""#### Giảng viên hướng dẫn: """)
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying spam and ham messages is one of the most common natural language processing tasks for emails and chat engines. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for ham and spam message classification.""")

elif choice == 'Build Project':
    

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = count_model.transform(lines)        
            y_pred_new = svc.predict(x_new)       
            st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new)) 

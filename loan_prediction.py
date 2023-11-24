#import all the important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import streamlit as st
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="LOAN APPROVAL SYSTEM",page_icon=":white_check_mark:",layout="wide")
st.title(":white_check_mark: LoanApproval_System")
st.markdown("<style>div.block-container{padding-top: 1 rem}</style>",unsafe_allow_html=True)


os.chdir(r"C:\Users\anshu\OneDrive\Desktop\python projects\loan approval system")
df = pd.read_csv("loan.csv",encoding="ISO-8859-1")

id = st.text_input("Enter Your Loan ID")
st.write(id)
loan_id = df["Loan_ID"]

listString=",".join(loan_id)



for i in loan_id:
         if i == str(id):
             st.success(f"{id} is present in the list!")
             selected_row = df[df["Loan_ID"]==id]
             st.write("Your Details:",selected_row)
             
             

col1, col2 = st.columns((2))


with col1:
    st.header("Loan Amount Graph")
    df['loanAmount_log']=np.log(df['LoanAmount'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['loanAmount_log'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of loanAmount_log')
    ax.set_xlabel('loanAmount_log')
    ax.set_ylabel('Frequency')


    st.pyplot(fig)
   
             
with col2:
    st.header("Total-Income Graph")
    df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
    df['TotalIncome_log']=np.log(df['TotalIncome'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['TotalIncome_log'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of TotalIncome_log')
    ax.set_xlabel('TotalIncome_log')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)
    
    
df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)
df['Married'].fillna(df['Married'].mode()[0],inplace =True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace =True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()

x= df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values


st.header("People Who Take loan Group by")
chart1,chart2 =st.columns((2))

with chart1:
    
    st.header(df['Gender'].value_counts())
    
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)        
    
    
with chart2:
    
    st.header(df['Married'].value_counts())
    
    fig, ax = plt.subplots()
    sns.countplot(x='Married', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)    
    
    
chart1,chart2 =st.columns((2))

with chart1:
    
    st.header(df['Dependents'].value_counts())
    
    fig, ax = plt.subplots()
    sns.countplot(x='Dependents', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)        
    
    
with chart2:
    
    st.header(df['Self_Employed'].value_counts())
    fig, ax = plt.subplots()
    sns.countplot(x='Self_Employed', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)    
        
st.header(df['LoanAmount'].value_counts())
fig, ax = plt.subplots()
sns.countplot(x='LoanAmount', data=df, palette='Set1', ax=ax)
st.pyplot(fig)  



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 0)

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()

for i in range(0,5):
  x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])
  x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])


labelencoder_y = LabelEncoder()
y_train =labelencoder_y.fit_transform(y_train)

labelencoder_y = LabelEncoder()
y_train =labelencoder_y.fit_transform(y_train)


for i in range(0,5):
  x_test[:,i] = labelencoder_x.fit_transform(x_test[:,i])
  x_test[:,7] = labelencoder_x.fit_transform(x_test[:,7])
  
  
labelencoder_y = LabelEncoder()

y_test = labelencoder_y.fit_transform(y_test)

from sklearn.preprocessing import StandardScaler

ss =  StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier

rf_clf =  RandomForestClassifier()
rf_clf.fit(x_train,y_train)

from sklearn import metrics
y_pred = rf_clf.predict(x_test)
        
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)

y_pred = nb_clf.predict(x_test)



from sklearn.tree import DecisionTreeClassifier
dt_clf =DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)

y_pred = dt_clf.predict(x_test)

from sklearn.neighbors import KNeighborsClassifier
kn_clf =KNeighborsClassifier()
kn_clf.fit(x_train,y_train)
y_pred = kn_clf.predict(x_test)

        


    

    
    
   
         
    
    
        
             
    
        






































































































































































    
    

 



   


    

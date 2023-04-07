import pandas as pd
import numpy as np
import sklearn
import streamlit as st
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
data=pd.read_excel("percentile_data.xlsx")
df=data.copy()
st.title("College Predictor")

def create_regressor(df):
    X=df['PERCENTILE'].values.reshape(-1,1)
    Y=df['RANK'].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    return regressor

def pvr(percentile,pwd,category):
    x=pd.Series([percentile])
    z=regressors[category][pwd=='YES'].predict(x.values.reshape(-1,1))
    k=float(np.round(z))
    if(k<=0):
        k=15
    return k

def output2(quota,category,gender,ranks):
    data1=pd.read_excel("college_data2.xlsx")
    df1=data1.copy()
    qts=df1[df1['Quota']==quota]
    ctg=qts[qts["Seat Type"]==category]
    gndr=ctg[ctg["Gender"]==gender]
    rnk=gndr[gndr["Closing Rank"]>=ranks]
    rnk1=rnk[rnk["Opening Rank"]<=ranks]
    x = rnk1.drop(['Quota','Seat Type','Gender','Opening Rank','Closing Rank'],axis=1).drop_duplicates()
    x.reset_index(inplace = True, drop = True)
    #x=x.head()
    return x
    
category=st.selectbox("Category : ",['OPEN','OBC-NCL','EWS', 'SC','ST'])
pwd=st.selectbox("PWD : ",['NO','YES'])

categories = ['OPEN', 'EWS', 'SC', 'ST', 'OBC-NCL']
regressors = {
        category : [
            create_regressor(df[df['CATEGORY']==category]),
            create_regressor(df[df['CATEGORY']==category +' '+ '(PwD)'])
            ] for category in categories
        }

percentile=st.number_input("Percentile : ",format="%.2f")

if(percentile == ""):
    st.text("Please Enter your Percentile")
else:
     ranks = int(pvr(float(percentile),pwd,category))

gender= st.selectbox("Gender : ",['Gender-Neutral', 'Female-only (including Supernumerary)'])
quota=st.selectbox("Quota : ",["OS","HS","AI"])

if st.button('Predict'):
    st.write("Your Category Rank is:",ranks)
    x=output2(quota,category,gender,ranks)
    if(x.empty):
        st.write("Sorry! no any college")
    else:
        st.dataframe(x)
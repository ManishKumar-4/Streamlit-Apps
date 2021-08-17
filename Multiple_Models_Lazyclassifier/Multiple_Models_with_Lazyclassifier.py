
# https://towardsdatascience.com/how-to-run-30-machine-learning-models-with-2-lines-of-code-d0f94a537e52

#Importing libraries
import pyforest
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(persist=True)
def split(df,test_size,target,predictors):
    y = df[target]
    x = df[predictors]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=0)
    return x_train, x_test, y_train, y_test

# It will help in storing dataset and do not read again when any change is done on UI
#@st.cache(allow_output_mutation=True)
@st.cache(persist=True)
def load_data(file):
    data = pd.read_csv(file)
    return data
    
@st.cache(persist=True)    
def preprocess_data(data):
    #Missing Value imputation and encoding categorical columns
    for col in data.columns:
        if data[col].dtypes=='object':
            data[col]=data[col].fillna(data[col].mode()[0])
            label = LabelEncoder()
            data[col] = label.fit_transform(data[col])
        else:
            data[col]=data[col].fillna(data[col].median())
    return data    
   

def main():

    #Seeing some of the data points
    
    st.title('Streamlit app for running different Machine learning models')
    st.sidebar.header('User Inputs')
    uploaded_file=st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'],key='uploaded_file')
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        data_1=data.copy()
               
        data_1=preprocess_data(data_1)
        if st.sidebar.checkbox("Show raw data",False):
            st.write(data)
            st.write('Above is raw data of shape:',data.shape)
       
        #EDA Purpose
        if st.sidebar.checkbox('Click the button for EDA',False):
            
            
            st.write('Missing Values present in the dataset:',data.isnull().sum())
            st.write('Descriptive Statistics:',data.describe())
            #st.write(data.describe(include=['O']))
            data.hist(figsize=(25,25))
            plt.show()
            st.pyplot()
            
            #Categorical column value counts
            cat_col=[]
            for col in data.columns:
                if data[col].dtypes=='object':
                    cat_col.append(col)
            #st.write('Categorical columns present in the dataset',cat_col)
            for i in cat_col:
                st.write(data[i].value_counts())
        
        if st.sidebar.checkbox("Show Preproces data",False):
           st.write(data_1)
           st.write('Above is Data After Preprocessing')
       
        st.sidebar.subheader("Features Selection & Data Partition")
        #if st.sidebar.checkbox("Train/Test Split (default 70:30)",False,key='t_t_split') :
        tt_split = st.sidebar.expander("Train/Test Split")
        target = tt_split.selectbox("Select Target Variable",data_1.columns,key="target")
        predictors = [v for v in data_1.columns if v!=target]
        new_predictors = tt_split.multiselect("Select Predictors",options=predictors,default=predictors)
        test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
        class_names = data_1[target].unique()
               
        if tt_split.checkbox("Dataset with selected features",False):
           st.write(data_1[new_predictors])
           st.write("Above is the dataset with selected features")
            
        if  tt_split.checkbox("Split the dataset",False):
            X_train, X_test, y_train, y_test = split(data_1,test_size,target,new_predictors)
            st.write('X Train Data shape after splitting',X_train.shape)
            st.write('X Test Data shape after splitting',X_test.shape)
               
            st.sidebar.subheader("Model Development")
            
            if st.sidebar.button("Different Models Result"):
            
                #Lazy classifier fitting
                clf = LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
                models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                st.write(models)
    
    
if __name__=='__main__':
    main()

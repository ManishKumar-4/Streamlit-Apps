#importing libraries
import streamlit as st
import mglearn
import graphviz

from sklearn.model_selection import train_test_split

import numpy as np
import time
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

from sklearn.metrics import mean_squared_error 
import math



@st.cache(persist=True) # It will help in storing dataset and do not read again when any change is done on UI
def load_data(file):
    data = pd.read_csv(file)
    label = LabelEncoder() # transform categorical to numaric
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(df,test_size,target,predictors):
    y = df[target]
    x = df[predictors]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=0)
    return x_train, x_test, y_train, y_test


#Function to simulate dummy dataset for classification problem

def simulate_data_classification():
    # simulate a dummy dataset 
    X, y = mglearn.datasets.make_forge()
    print("X.shape:", X.shape)

    # see data pattern
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    # plot parms
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()
    
#Function to simulate dummy dataset for regression problem

def simulate_data_regression():
    X, y = mglearn.datasets.make_wave(n_samples=40)
    test_data = [-1.5, 0.9, 1.5]

    # view data scatterplot pattern
    plt.plot(X, y, 'bo', label="data")
    plt.plot(test_data, [-2.7, -2.7, -2.7], 'g*', label="test points")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()


#function to build kNN using mglearn inbuilt figures.

def kNN_mglearn_classification(n_neighbors):
    return mglearn.plots.plot_knn_classification(n_neighbors=n_neighbors)
    
  
def kNN_mglearn_regression(n_neighbors):
    return mglearn.plots.plot_knn_regression(n_neighbors=n_neighbors)
    
    

#Function to split the data and to fit the model

def data_split_classification(test_size,n):
    X, y = mglearn.datasets.make_forge()
    #stratified train-test split with random seed
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
    
    ## instantiate model and training model on train data
    
    clf = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)  
    return X_train,X_test,y_train,y_test,clf
    
 
def data_split_regression(test_size):
    X, y = mglearn.datasets.make_wave(n_samples=40)

    # split the wave dataset into a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size,random_state=0)
    return X_train,X_test,y_train,y_test
    
#Function to check accuracy and plot for the kNN classification model

def kNN_classification_pred_plot(test_size,n):
    X_train,X_test,y_train,y_test,clf=data_split_classification(test_size,n)
    predictions=clf.predict(X_test)
    
    # plotting train & test pts separately
    y_train_0 = (y_train == 0)  # make indices
    y_train_1 = (y_train == 1)

    y_test_0 = (predictions == 0)
    y_test_1 = (predictions == 1)

    plt.plot(X_train[y_train_0, 0], X_train[y_train_0, 1], 'bo', label="Train 0")
    plt.plot(X_train[y_train_1, 0], X_train[y_train_1, 1], 'g^', label="Train 1")

    plt.plot(X_test[y_test_0, 0], X_test[y_test_0, 1], 'ro', label="Test 0")
    plt.plot(X_test[y_test_1, 0], X_test[y_test_1, 1], 'r^', label="Test 1")

    plt.legend()
    
    
def plot_regression(test_size,n):
    # setting stage to show 3 (sub)plots side-by-side
    X_train, X_test, y_train, y_test = data_split_regression(test_size)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # create 1,000 data points, evenly spaced between -3 and 3
    import numpy as np
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)


    # loop over k=1,3,9 and over axes
    for n_neighbors, ax in zip([n, n+2, n+8], axes):

        # make predictions using 1, 3, or 9 neighbors
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)

        # plot and show	
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

        ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))

        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")


    axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
    

def main():
    
    st.set_option('deprecation.showPyplotGlobalUse',False)
    st.title('Streamlit kNN app for classification and Regression')
    st.write('Aim is to demo basic concepts.Select the option given on the left to proceed:')
    st.sidebar.subheader('User Inputs')
    problem_type=st.sidebar.radio('Problem Type',("Classification","Regression"),key = 'problem_type')
    
    if problem_type=='Classification':
        menu=["Dummy Data Plot","Different n plot with Dummy Data","kNN Classification with Dummy Data","kNN on Real dataset"]
        #menu=["Show Dummy data","Different n plot","kNN Classification","kNN Regression","kNN Regression different n","kNN on Real dataset"]
        choice=st.sidebar.selectbox("Menu",menu)
    
        if choice=="Dummy Data Plot":
            st.markdown("### Simulated dataset plot")
            #st.write('How many data points to simulate in 2D')
            st.pyplot(simulate_data_classification())
        
        if choice=="Different n plot with Dummy Data":
            st.markdown("### Plot with different neighbors")
            st.write('How many neighbors you want to consider')
            n=st.sidebar.slider("Select the number of neighbors",1,5)
            st.text('Selected: {}'.format(n))
            st.pyplot(kNN_mglearn_classification(n))
            st.write('The stars in the graph above are the unlabeled units for which we have to predict the class.')
        
        if choice=="kNN Classification with Dummy Data":
            st.markdown(" ## Accuracy and plot for kNN classification")
            #st.sidebar.subheader("Data Partition")
            st.sidebar.subheader("Data Partition")
            tt_split = st.sidebar.beta_expander("Train/Test Split")
            test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
            n=st.sidebar.slider("Select the number of neighbors",1,5)
            
            X_train,X_test,y_train,y_test,clf=data_split_classification(test_size,n)
            
            predictions=clf.predict(X_test)
            #st.write("Specs of the kNN: ", clf, "\n")
    
            # apply trained model on test
            st.markdown(" ## Evaluating kNN Performance")
            st.write("kNN training set score:",round(clf.score(X_train,y_train),2),"\n")
            #st.write("kNN test set prediction:",predictions)
            st.write("kNN test set score:",round(clf.score(X_test,y_test),2),"\n")
            st.pyplot(kNN_classification_pred_plot(test_size,n))
            st.write('The red points are test data. Their shape (triangle vs circle) tells us of the class they were assigned to.')
        
        if choice=="kNN on Real dataset":
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'],key='uploaded_file')
            if uploaded_file is not None:
               data = load_data(uploaded_file)
               if st.sidebar.checkbox("Show raw data",False):
                  st.write(data)
               st.sidebar.subheader("Data Partition")
               #if st.sidebar.checkbox("Train/Test Split (default 70:30)",False,key='t_t_split') :
               tt_split = st.sidebar.beta_expander("Train/Test Split")
               target = tt_split.selectbox("Select Target Variable",data.columns,key="target")
               predictors = [v for v in data.columns if v!=target]
               new_predictors = tt_split.multiselect("Select Predictors",options=predictors,default=predictors)
               test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
               class_names = data[target].unique()
               
            #X_train, X_test, y_train, y_test = split(data,test_size,target,new_predictors)
               if  tt_split.button("split",key = "split"):
                    X_train, X_test, y_train, y_test = split(data,test_size,target,new_predictors)
               else:
                    X_train, X_test, y_train, y_test = split(data,0.30,target,new_predictors)
                    
               st.write('X Train Data shape after splitting',X_train.shape)
               st.write('X Test Data shape after splitting',X_test.shape)
            
               st.sidebar.subheader("Model Development")
     
               n=st.sidebar.slider("Select the number of neighbors for kNN",1,10)
               
               st.sidebar.write('Click the below Plot button after selecting a particular n value')
               if st.sidebar.button("Plot"):
                # deine empty lists to capture output
                training_accuracy = []
                test_accuracy = []
                
                # try n_neighbors from 1 to 10
                neighbors_settings = range(1, n+1)
                
                # use a for loop over k=1 to 10
                for n_neighbors in neighbors_settings:

                    # build the model
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                    clf.fit(X_train, y_train)

                    # record training set accuracy
                    training_accuracy.append(clf.score(X_train, y_train))
    
                    # record generalization accuracy
                    test_accuracy.append(clf.score(X_test, y_test))
                    
                # now plot the results and see    
                plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
                plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
                plt.ylabel("Accuracy")
                plt.xlabel("n_neighbors")
                plt.legend()
                st.pyplot()
   
    if problem_type=='Regression':
    
        menu=["Dummy data plot","kNN on Dummy Set with different n","kNN on Dummy Set","kNN Regression on Real Dataset"]
        choice=st.sidebar.selectbox("Menu",menu)
    
        if choice=="Dummy data plot":
            st.markdown("### Simulated Dataset Plot")
            #st.write('How many data points to simulate in 2D')
            st.pyplot(simulate_data_regression())
            st.write('In the above figure Vertical axis is target or Y variable. Horiz axis is a feature or a (say) 1-D projection of feature set.Given locations of new data (green stars) on the X-axis, what is their predicted Y value')
        
        if choice=="kNN on Dummy Set with different n":
            st.markdown("### Plot with different neighbors")
            st.write('How many neighbors you want to consider')
            n=st.sidebar.slider("Select the number of neighbors",1,5,value=1)
            st.text('Selected: {}'.format(n))
            st.pyplot(kNN_mglearn_regression(n))
            st.write('From the above figure Under kNN(k=1), the predicted Y is merely the same Y as that of the nearest neighbor on the x-axis to the new data point.Same carries over as weighted mean Y of k nearest neighbors when analyzed with kNN(k=k).')
        
        if choice=="kNN on Dummy Set":
            
            st.markdown(" ## Accuracy and plot for kNN regression")
            st.sidebar.subheader("Data Partition")
            tt_split = st.sidebar.beta_expander("Train/Test Split")
            test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
            n=st.sidebar.slider("Select the number of neighbors",1,5)
            X_train, X_test, y_train, y_test = data_split_regression(test_size)

            # instantiate the model and set the number of neighbors to consider to 3
            reg = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)

            st.write("kNN test set predictions:\n",reg.predict(X_test),"\n")
            st.write("Test set R^2: ",round(reg.score(X_test,y_test),2),"\n")
        
            st.pyplot(plot_regression(test_size,n))
            st.write("Note how the regression line (blue line connecting the blue triangle data) smoothens out and flattens out as k rises")

            st.write("Note also how the training and test score (akin to R^2) changes with k")

            st.write("At k=1 overfits the data (hence training RMSE=0 & score=1) but generalizes very poorly to unseen test data (score is merely 0.35)")

            st.write("Seems, k=3 is ideal with test score actually beating even training score")
            
            
        if choice=="kNN Regression on Real Dataset":
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'],key='uploaded_file')
            if uploaded_file is not None:
               data = load_data(uploaded_file)
               if st.sidebar.checkbox("Show raw data",False):
                  st.write(data)
               st.sidebar.subheader("Data Partition")
               #if st.sidebar.checkbox("Train/Test Split (default 70:30)",False,key='t_t_split') :
               tt_split = st.sidebar.beta_expander("Train/Test Split")
               target = tt_split.selectbox("Select Target Variable",data.columns,key="target")
               predictors = [v for v in data.columns if v!=target]
               new_predictors = tt_split.multiselect("Select Predictors",options=predictors,default=predictors)
               test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
               class_names = data[target].unique()
               
            #X_train, X_test, y_train, y_test = split(data,test_size,target,new_predictors)
               if  tt_split.button("split",key = "split"):
                    X_train, X_test, y_train, y_test = split(data,test_size,target,new_predictors)
               else:
                    X_train, X_test, y_train, y_test = split(data,0.30,target,new_predictors)
                    
               st.write('X Train Data shape after splitting',X_train.shape)
               st.write('X Test Data shape after splitting',X_test.shape)
            
               st.sidebar.subheader("Model Development")
     
               n=st.sidebar.slider("Select the number of neighbors for kNN",1,10)
               
               st.sidebar.write('Click the below Plot button after selecting a particular n value')
               if st.sidebar.button("Plot"):
                # deine empty lists to capture output
                
                rmse_values=[]
                training_accuracy = []
                test_accuracy = []
                
                # try n_neighbors from 1 to 10
                neighbors_settings = range(1, n+1)
                
                # use a for loop over k=1 to 10
                for n_neighbors in neighbors_settings:

                    # build the model
                    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
                    reg.fit(X_train, y_train)
                    
                    y_pred=reg.predict(X_test)
                    
                    err=math.sqrt(mean_squared_error(y_test,y_pred))
                    
                    rmse_values.append(err)
                    # record training set accuracy
                    training_accuracy.append(reg.score(X_train, y_train))
    
                    # record generalization accuracy
                    test_accuracy.append(reg.score(X_test, y_test))
                    
                
                    
                # now plot the results and see 
                plt.subplot(2, 1, 1)
                plt.plot(neighbors_settings, rmse_values, label="RMSE")
                plt.ylabel("RMSE")
                plt.xlabel("n_neighbors")
                
                plt.subplot(2, 1, 2)
                # now plot the results and see    
                plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
                plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
                plt.ylabel("Accuracy")
                plt.xlabel("n_neighbors")
                plt.legend()
                st.pyplot()
            
            
       
if __name__=='__main__':
    main()
#importing libraries

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
import time
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier  # Dtree classifn
from sklearn.tree import DecisionTreeRegressor   # Dtree regressn
from sklearn.linear_model import LinearRegression  # for comparison
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons  # to simulate dummy data
# plotting
#%matplotlib inline
#from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import graphviz
import mglearn

from matplotlib.colors import ListedColormap

from IPython.display import clear_output, Image, display


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



def main():

    def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], legend=False, plot_training=True):
    
        x1s = np.linspace(axes[0], axes[1], 100)  # akin to seq(start, stop, num_breaks)
        x2s = np.linspace(axes[2], axes[3], 100)
        x1, x2 = np.meshgrid(x1s, x2s)
    
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
    
        # plot contour maps, select colors from custom palettes
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    
        if plot_training:
            plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
            plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
            plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
            plt.axis(axes)
        
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
        if legend:
            plt.legend(loc="lower right", fontsize=14)

    st.set_option('deprecation.showPyplotGlobalUse',False)
    st.title('Streamlit App for Decision Tree Classifier')
    st.write('For performing different operations use left side panel')
    st.sidebar.header('User Input')
    menu=['Decision Tree-Dummy Dataset','Rotating Data&Tree Performance','Real World Dataset']
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice=='Decision Tree-Dummy Dataset':
        n_samples=st.sidebar.slider('Select the number of samples',100,1000)
        noise=st.sidebar.slider('Select the noise',0.1,0.5)
        
        Xm, ym = make_moons(n_samples=n_samples, noise=noise, random_state=53)
        st.write("Xm sample:\n", Xm[:5,:])
        
        #print("\n")
        st.write("ym sample: ", ym[:5])
        st.write('Above is shown only five samples out of n_samples =',n_samples)
        plt.scatter(Xm[:,0], Xm[:,1], c=ym)
        st.pyplot()
        st.write('Above figure is scatter plot for the given samples')
        
        # fit an UNRESTRICTED D-Tree classifiers on it
        fitting_type=st.sidebar.radio('Model fitting',['Unrestricted Tree','Restricted Tree'],key = 'fitting_type')
        
        if fitting_type=='Unrestricted Tree':
        
            st.subheader('Let us see when Decision Tree is fitted on the sample dataset')
            deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
            deep_tree_clf1.fit(Xm, ym)
            st.write("Accuracy on Unrestricted Tree: ", round(deep_tree_clf1.score(Xm, ym), 3))
            plt.figure(figsize=(11, 8))
            #plt.subplot(121)
            plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5])
            plt.title("No restrictions", fontsize=16)
            st.pyplot()
            st.write('The above figure shows Decision Tree boundary for unrestricted tree')
            
        if fitting_type=='Restricted Tree':
            # D-tree below restricts each leaf node to have 4+ instances
            min_samples_leaf=st.sidebar.slider('Minimum samples leaf',2,10)
            
            deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
            deep_tree_clf2.fit(Xm, ym)
            st.write("Accuracy on Restricted Tree:", round(deep_tree_clf2.score(Xm, ym), 3)) 
    
            plt.figure(figsize=(11, 8))
            #plt.subplot(121)
            
            plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5])
            plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
            plt.show()
            st.pyplot()
            st.write('The above figure shows Decision Tree boundary for restricted tree')
            
            
            ## build D-tree itself and see
            plt.subplot(122)
            export_graphviz(deep_tree_clf2,    # fitted model here
                    out_file="tree.dot", 
                    class_names=["Purple", "Yellow"],  # user input here
                    feature_names = ['X1', 'X2'],      # user input here
                    impurity=False, filled=True)

            # display the tree with graphviz
            with open("tree.dot") as f:
                dot_graph = f.read()
            st.graphviz_chart(dot_graph)

    
    if choice=='Rotating Data&Tree Performance':
            
        ## Concept: sensitivity to rotation of tree methods

        # simulating dummy data to demo the concept
        np.random.seed(6)
        Xs = np.random.rand(100, 2) - 0.5
        st.write("Xs vals: \n", Xs[:8,:])
        ys = (Xs[:, 0] > 0).astype(np.float32) * 2
        st.write("ys vals: ", ys[:8])
        st.write('Above data are some samples of complete dummy dataset')
            
        # view data pattern
        plt.scatter(Xs[:,0], Xs[:,1], c=ys)
        st.pyplot()
        st.write('Above is shown Scatter plot without any rotation')
        # 'rotate' the current data representation
        angle=st.sidebar.select_slider('Angle',[0,15,30,45,60,90,105,120,135,150,165,180],value=45)
        #angle = np.pi / 4
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        Xsr = Xs.dot(rotation_matrix)

        plt.scatter(Xsr[:,0], Xsr[:,1], c=ys)  # view rotated representation
        st.pyplot()
        st.write('Above is shown Scatter plot with rotation of',angle,'degree')
        # now fit DTree on both original & rotated data representations
        tree_clf_s = DecisionTreeClassifier(random_state=42)
        tree_clf_s.fit(Xs, ys)

        tree_clf_sr = DecisionTreeClassifier(random_state=42)
        tree_clf_sr.fit(Xsr, ys)
        
        
        # Now plot DTree performance on both representations
        plt.figure(figsize=(11, 4))
        plt.subplot(121)
        plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7])
        plt.subplot(122)
        plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7])
        st.pyplot()
        st.write('The above figure shows fitting of decision boundary in non rotated and',angle,'degree rotated case')
        
        
    if choice=='Real World Dataset':
        #st.sidebar.write('Upload the dataset')
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'],key='uploaded_file')
        if uploaded_file is not None:
            data=load_data(uploaded_file)
            st.write('Shape of the dataset',data.shape)
            st.write(data.head(10))
            st.sidebar.subheader("Data Partition")
            tt_split = st.sidebar.beta_expander("Train/Test Split")
            target=tt_split.selectbox("Select Target Variable",data.columns,key="target")
            predictors=[v for v in data.columns if v!=target]
            new_predictors=tt_split.multiselect("Select Predictors",options=predictors,default=predictors)
            test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.5,step=0.1,key="test_size",value=0.30)
            class_names = data[target].unique()
            
            if  tt_split.button("Press for Splitting the dataset",key = "split"):
                X_train, X_test, y_train, y_test = split(data,test_size,target,new_predictors)
                
            else:
                X_train, X_test, y_train, y_test = split(data,0.3,target,new_predictors)
                
            st.write('X Train Data shape after splitting',X_train.shape)
            st.write('X Test Data shape after splitting',X_test.shape)
                
            st.sidebar.subheader('Model Development')
            max_depth=st.sidebar.slider('Maximum depth',1,10)
            
            # prune tree with 'max_depth' against overfitting & for better generalizn
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
            tree.fit(X_train, y_train)
            st.write("Accuracy on training set: ", round(tree.score(X_train, y_train), 3))   
            st.write("Accuracy on test set: ", round(tree.score(X_test, y_test), 3)) 

            ## Analyzing Decision Trees
            
            st.subheader('Analyzing the Decision Trees')
            st.sidebar.write("Click the button below after setting the depth")
            if st.sidebar.button('Analyze the Decision Tree'):
                export_graphviz(tree, 
                    out_file="tree.dot", 
                    class_names=["survived", "died"],
                    feature_names = predictors, impurity=False, filled=True)

            # display the tree with graphviz
                with open("tree.dot") as f:
                    dot_graph = f.read()
                st.graphviz_chart(dot_graph)
                #Feature importance and plot
                st.subheader('Feature Importance')
                feat_imp_df = pd.DataFrame({'variable': predictors, 'imp_score':tree.feature_importances_})
                st.write(feat_imp_df.sort_values(by = 'imp_score', ascending=False))
                
                
                #feature importance plot
                st.subheader('Feature Importance Plot')
                n_features = data.shape[1]-1
                plt.barh(np.arange(n_features), tree.feature_importances_, align='center')
                plt.yticks(np.arange(n_features), predictors)
                plt.xlabel("Feature importance")
                plt.ylabel("Feature")
                plt.ylim(-1, n_features)
                st.pyplot()
            
            

    
if __name__=='__main__':
    main()
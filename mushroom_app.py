import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score,recall_score
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?  ")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?  ")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        pass
        # '''
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            try:
                from sklearn.metrics import plot_confusion_matrix
                plot_confusion_matrix(model, x_test, y_test,display_labels=class_names)
            except:
                from sklearn.metrics import classification_report
                confusion_mat=confusion_matrix(y_test,model.predict(x_test))
                confusion_mat=[confusion_mat[1],confusion_mat[0]]
                sns.heatmap(confusion_mat,cbar=False,annot=True,fmt='d',cmap=plt.cm.Blues)
                plt.xlim(0,2)
                plt.ylim(0,2)
                plt.xlabel('Actual')
                plt.ylabel('prediction')
                plt.xticks([0.5,1.5],labels=['edible','poisinous'])
                plt.yticks([0.5,1.5],labels=['poisinous','edible'])        
            st.pyplot()
        # '''
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            try:
                from sklearn.metrics import plot_roc_curve
                plot_roc_curve(model,X_test,y_test)
            except:    
                y_pred_proba=model.predict_proba(x_test)[:,1]
                fpr,tpr,_=roc_curve(y_test,y_pred_proba)
                auc = roc_auc_score(y_test,y_pred_proba)
                plt.plot(fpr,tpr,label="auc_score({})".format(auc.round(3)))
                plt.plot([0,1],[0,1],'b--')
                plt.grid()
                plt.legend()
                plt.xlabel('false positive rate (fpr)')
                plt.ylabel('true positive rate (tpr)')
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            try:
                from sklearn.metrics import plot_precision_recall_curve
                plot_precision_recall_curve(model,x_test,y_test)
            except:    
                y_pred_proba=model.predict_proba(x_test)[:,1]
                precision,recall,_=precision_recall_curve(y_test,y_pred_proba)
                plt.plot(recall,precision,label='precision-recal curve')
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.grid()
                plt.legend()
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic Regression","Random Forest"),key='classifier')

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparmeters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.1,key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"),key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")

        metrics=st.sidebar.multiselect("What matrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
        if(st.sidebar.button("Classify",'classify')):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.1,key='C')
        max_iter=st.sidebar.slider("Maximum number of iterations",100, 500,key='max_iter')

        metrics=st.sidebar.multiselect("What matrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')

        if(st.sidebar.button("Classify",'classify')):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter=max_iter,solver='lbfgs')
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in forest",100,5000,step=10,key='n_estimators')
        max_depth=st.sidebar.slider("The maximum depth of the tree",1, 20,key='max_depth')
        bootstrap=st.sidebar.radio("Bootstrap sample when building trees", ('True','False'),key='bootstrap')

        metrics=st.sidebar.multiselect("What matrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')

        if(st.sidebar.button("Classify",'classify')):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)            




    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df.head())

main()
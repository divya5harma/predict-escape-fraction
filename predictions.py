
# Using 30% threshold model for predictions
# total values = 1813

# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")

def model(df):
    df = df.sort_values('Escape_fraction',ascending=False)
    z = 0.30               # 30% threshold
    a = []
    b = []
    c = []
    d = []

    x = int(len(df)*z)
    y = int(len(df)-x)
    df1 = df.iloc[0:x,:]
    df2 = df.iloc[y:,:]
    df1['labels'] = 1
    df2['labels'] = 0

    dff = pd.concat([df1,df2]).reset_index()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dff[['Main_Chain_Side_Chain_Hydrogen_Bonds','Backbone_Hydrogen_bond_Energy','Electrostatics_Energy','vdW_Clash_Energy','ΔΔGmcsm']], dff['labels'], test_size=0.2, random_state=42)

    # Randomforest classifier
    clf = RandomForestClassifier(n_estimators=100,random_state=42)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Predict the labels of the test set using the trained classifier
    y_pred = clf.predict(X_test)
    
    # Calculate the accuracy of the classifier on the test set
    accuracy = accuracy_score(y_test, y_pred)

    #Creating the Confusion matrix  

    cm= confusion_matrix(y_test, y_pred)  

    acc = accuracy_score(y_test, y_pred)
    # Sensitivity
    sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
    sensitivity1 = round(sensitivity1,2)
    # Specificity
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    specificity1 = round(specificity1,2)
    # Balanced acuuracy
    bal_acc=balanced_accuracy_score(y_test,y_pred)
    bal_acc = round(bal_acc,2)

    # Get predicted probabilities of positive class for test set
    y_score = clf.predict_proba(X_test)[:,1]

    # Compute roc_auc_score for the classifier
    auc = roc_auc_score(y_test, y_score)
    auc = round(auc,2)
    # Append all values to list
    a.append(bal_acc)
    b.append(sensitivity1)
    c.append(specificity1)
    d.append(auc)
    
    svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
    line = svc_disp.line_
    line.set_color('black')
    svc_disp.ax_.set_xlabel("False Positive Rate", fontsize=12, labelpad=5)
    svc_disp.ax_.set_ylabel("True Positive Rate", fontsize=12, labelpad=5)
    svc_disp.ax_.legend(handles=[line], labels=['RandomForestClassifier (AUC = 0.91)'], facecolor='white',loc='lower right')
    
    # Prepare a dataframe with the results
    dfz = pd.DataFrame({'threshold':z,'bal_acc':a,'sensitivity':b,'specificity':c,'auc':d})
    
    return clf,dfz

def predictions(dfj):
    # Load the test set data without labels
    X_test = dfj
    # Use the trained model to predict the labels for the test set
    y_pred = classifier.predict(X_test)

    # Get the predicted labels for the test set
    predicted_probs = classifier.predict_proba(X_test)             # get predicted probabilities of the prediction
    dg = pd.DataFrame({'escape':y_pred})
    dg['probability'] = 'probs'

    # get predicted probabilities
    for i in range(len(dg)):
        dg['probability'][i] = predicted_probs[i]     # predicted_probs contains a list of predicted probability of class 0, class 1

    # Concat the two dataframes to get entire predicted data
    df_predict = pd.concat([dfj,dg],axis=1).reset_index()
    
    return df_predict

# Read csv file containing escape fraction data
dfl = pd.read_csv('Dataset1.csv')

# Call the function
classifier, evaluation = model(dfl)
print('Model performance:\n')
print(evaluation)
print('\n')

# Predict high and low escape fraction for mutation data
# dfm = pd.read_excel('updated_reanalysis/exp_mutations.xlsx')
dfm = pd.read_csv('data_for_prediction.csv')
dfm = dfm[['Main_Chain_Side_Chain_Hydrogen_Bonds','Backbone_Hydrogen_bond_Energy','Electrostatics_Energy','vdW_Clash_Energy','ΔΔGmcsm']]
result_pred = predictions(dfm)
print('Predicted results:\n')
print(result_pred)
# If you want to save the final predicted results as csv file, uncomment the below line
# result_pred.to_csv('final_predcitions.csv',index=False)

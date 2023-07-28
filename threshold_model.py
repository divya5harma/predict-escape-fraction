# results for different thresholds
## Code for model development at different thresholds
# total mutations = 1813

# Import librarires
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
pd.options.mode.chained_assignment = None       # No warnings

def model_thresholds(df):
    df = df.sort_values('Escape_fraction',ascending=False)

    z = [0.3,0.25,0.20,0.15,0.1,0.05]
    a = []
    b = []
    c = []
    d = []
    for i in z:
        x = int(len(df)*i)
        y = int(len(df)-x)
        df1 = df.iloc[0:x,:]
        df2 = df.iloc[y:,:]
        df1['labels'] = 1
        df2['labels'] = 0

        dff = pd.concat([df1,df2]).reset_index()

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(dff[['Main_Chain_Side_Chain_Hydrogen_Bonds','Backbone_Hydrogen_bond_Energy','Electrostatics_Energy','vdW_Clash_Energy','ΔΔGmcsm']], dff['labels'], test_size=0.2, random_state=42)

        # Initialize the Randomforest classifier
        clf = RandomForestClassifier(n_estimators=100,random_state=42)

        # Fit the Randomforest classifier to the training data
        clf.fit(X_train, y_train)

        # Predict the labels of the test set using the trained classifier
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
        # Balanced accuracy
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

    # Prepare a dataframe with the results
    dfz = pd.DataFrame({'threshold':z,'bal_acc':a,'sensitivity':b,'specificity':c,'auc':d})
    return dfz

# Read csv file containing escape fraction data
dfl= pd.read_csv('Dataset1.csv')
dfl = dfl[dfl['Escape_fraction'].notna()]

print(len(dfl))
# Call the function
threshold = model_thresholds(dfl)
print(threshold)

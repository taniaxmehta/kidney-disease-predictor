'''
Created on Aug. 5, 2023
used on command line: python kidney_disease_project.py
Install Libraries:
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install numpy

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.pyplot import xlabel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

def isfloat(num):
    '''
    for checking input in data_analysis()
    '''
    try:
        float(num)
        return True
    except ValueError:
        return False

def select_menu():
    '''
    This is the main menu
    Job: selects the algorithm or exit option
    '''
    print("Select an option:")
    print("1- Exploratory Data Analysis")
    print("2- Logistic Regression")
    print("3- Exit")
    #Start
    option = input('Option number: ')
    while((not option.isdigit()) or int(option)<1 or int(option)>3):
        print("<Not a valid option number>")
        option = input('Option number: ')
    return int(option)

def select_analysis():
    '''
    This is the menu for data_analysis()
    Job: selects the options
    '''
    print("Select an option:")
    print("1- CKD cases count")
    print("2- missing values Column-Wise")
    print("3- category graph observations")
    print("4- Correlation Matrix")
    print("5- Exit")
    #Start
    option = input('Option number: ')
    while((not option.isdigit()) or int(option)<1 or int(option)>5):
        print("<Not a valid option number>")
        option = input('Option number: ')
    return int(option)

def select_input():
    '''
    This is the input menu for data_analysis()
    Job: selects the input or exit option
    '''
    print("Select an option:")
    print("1- Inputs and Prediction")
    print("2- Sample Prediction")
    print("3- Exit")
    #Start
    option = input('Option number: ')
    while((not option.isdigit()) or int(option)<1 or int(option)>3):
        print("<Not a valid option number>")
        option = input('Option number: ')
    return int(option)

def data_analysis(df):
    #Start program
    option = 0
    while(option != 5):
        option = select_analysis()
        print("")
        if(option == 1):
            print("<show CKD cases count in dataset>")
            df.target.value_counts().plot(kind="bar", color=["salmon", "lightgreen"], xlabel="Have CKD",
                                          ylabel="Cases")
            plt.title("Chronic Kidney Disease cases in Indian population")
            plt.show()
        elif(option == 2):
            print("<Missing values Column-Wise>")
            nan_count = df.isna().sum()
            print(nan_count)
        elif(option == 3):
            print("<show category graph observations>")
            categorical_val = []
            continous_val = []
            for column in df.columns:
                if len(df[column].unique()) < 10:
                    categorical_val.append(column)
                else:
                    continous_val.append(column)
            #print
            for i, column in enumerate(categorical_val, 1):
                CrosstabResult=pd.crosstab(index=df[column],columns=df['target'])
                CrosstabResult.plot.bar(figsize=(7,4), rot=0)
                plt.xlabel(column)
                plt.show()
            for i, column in enumerate(continous_val, 1):
                df[df["target"] == 0][column].hist(bins=30, color='blue', label='Have CKD = NO', alpha=0.6)
                df[df["target"] == 1][column].hist(bins=30, color='red', label='Have CKD = YES', alpha=0.6)
                plt.legend()
                plt.xlabel(column)
                plt.show()
        elif(option == 4):
            print("<Correlation Matrix>")
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(25, 25))
            ax = sns.heatmap(corr_matrix,annot=True,
                             linewidths=0.8,fmt=".2f",cmap="YlGnBu")
            plt.show()
            df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")
            plt.show()

def data_processing(df):
    #prepare dataset
    data_columns = []
    for column in df.columns:
        if(column != "target"):# and (len(df[column].unique()) < 10):
            data_columns.append(column)
    dataset = pd.get_dummies(df, columns = data_columns)
    s_sc = StandardScaler()
    col_to_scale = ['age', 'sg', 'al', 'pc', 'hemo', 'htn', 'dm']
    index_col = []
    for index in dataset.columns:
        for col in col_to_scale:
            x = index.find(col)
            if(x != -1): index_col.append(index) 
            #the columns from get_dummies() are split into results, so it appears as 'dm_0.0', 'dm_1.0'
    dataset[index_col] = s_sc.fit_transform(dataset[index_col])
    #train the learning model
    X = dataset.drop('target', axis=1)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    #Logistic Regression
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
    
    #inputs
    attribute_name = ["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc",
                      "sod","pot","hemo","pcv","wbcc","rbcc","htn","dm","cad","appet","pe","ane"]
    full_name = ["Age","Blood Pressure in mm/Hg","Specific Gravity (1.005,1.010,1.015,1.020,1.025)",
                 "Albumin (0,1,2,3,4,5)","Sugar (0,1,2,3,4,5)","Red Blood Cells (0:normal,1:abnormal)",
                 "Pus Cell (0:normal,1:abnormal)","Pus Cell clumps (1:present,0:not present)",
                 "Bacteria (1:present,0:not present)","Blood Glucose Random in mgs/dl","Blood Urea in mgs/dl",
                 "Serum Creatinine in mgs/dl","Sodium in mEq/L","Potassium in mEq/L","Hemoglobin in gms",
                 "Packed  Cell Volume","White Blood Cell Count in cells/cumm",
                 "Red Blood Cell Count in millions/cmm","Hypertension (1:yes,0:no)",
                 "Diabetes Mellitus (1:yes,0:no)","Coronary Artery Disease (1:yes,0:no)",
                 "Appetite (0:good,1:poor)","Pedal Edema (1:yes,0:no)","Anemia (1:yes,0:no)"]
    option = 0
    while(option != 3):
        option = select_input()
        print("")
        if(option == 1): #Get inputs and Prediction
            print("Input Patient Info")
            input_line = []
            for i in range(0,24):
                option = input(f'{full_name[i]}: ')
                if(i==0 or i==1 or (i>=9 and i<=17)):
                    while((not isfloat(option)) or float(option)<0):
                        print("<Not a valid number>")
                        option = input(f'{full_name[i]}: ') 
                    input_line.append(float(option))
                elif(i==2): #Specific Gravity (1.005,1.010,1.015,1.020,1.025)
                    while((not isfloat(option)) or (float(option)!=1.005 and float(option)!=1.010
                                                     and float(option)!=1.015 and float(option)!=1.020
                                                     and float(option)!=1.025)):
                        print("<Not a valid number>")
                        option = input(f'{full_name[i]}: ') 
                    input_line.append(float(option))
                elif(i==3 or i==4):
                    while((not option.isdigit()) or int(option)<0 or int(option)>5):
                        print("<Not a valid number>")
                        option = input(f'{full_name[i]}: ') 
                    input_line.append(int(option))
                else: #(0,1) options
                    while((not option.isdigit()) or int(option)<0 or int(option)>1):
                        print("<Not a valid number>")
                        option = input(f'{full_name[i]}: ') 
                    input_line.append(int(option))
            #Done inputs
            find_name = []
            for i in range(0,len(input_line)):
                name = f"{attribute_name[i]}_{input_line[i]}"
                if(name == "sg_1.010"): find_name.append("sg_1.01")
                elif(name == "sg_1.020"): find_name.append("sg_1.02")
                else: find_name.append(name)
            inner_array = []
            for col in dataset.columns:
                if(col != 'target'):
                    Found = False
                    for name in find_name:
                        x = col.find(name)
                        if(x != -1): 
                            inner_array.append(1)
                            Found = True
                    if(not Found):
                        inner_array.append(0)
            #setup inputs Done
            features = np.array([inner_array])
            # using inputs to predict the output
            warnings.filterwarnings("ignore")
            prediction = lr_clf.predict(features)
            probabilities = lr_clf.predict_proba(X_test)
            odds = max(probabilities[0]) * 100
            if(odds == 100): odds = 99.99       #nothing is guaranteed
            if(prediction[0]==1):
                print("Prediction: The patient has Chronic Kidney Disease(CKD)")
            else:
                print("Prediction: The patient does not have Chronic Kidney Disease(CKD)")
            print("Confidence: {:.2f}%".format(odds))
        elif(option == 2): #Sample Prediction
            print("Sample Patient Info")
            input_line = []
            print("1- positive sample")
            print("2- negative sample")
            #Start
            sample_num = input('Option number: ')
            while((not sample_num.isdigit()) or int(sample_num)<1 or int(sample_num)>2):
                print("<Not a valid option number>")
                sample_num = input('Option number: ')
            if(int(sample_num)==1):
                input_line = [68,80,1.010,3,2,0,1,1,1,157,90,4.1,130,6.4,5.6,16,11000,2.6,1,1,1,1,1,0] #copied result is 1
            else:
                input_line = [33,80,1.025,0,0,0,0,0,0,128,38,0.6,135,3.9,13.1,45,6200,4.5,0,0,0,0,0,0] #copied result is 0
            for i in range(0,24):
                print(f"{full_name[i]}: {input_line[i]}")
            #Done inputs
            find_name = []
            for i in range(0,len(input_line)):
                name = f"{attribute_name[i]}_{input_line[i]}"
                if(name == "sg_1.010"): find_name.append("sg_1.01")
                elif(name == "sg_1.020"): find_name.append("sg_1.02")
                else: find_name.append(name)
            inner_array = []
            for col in dataset.columns:
                if(col != 'target'):
                    Found = False
                    for name in find_name:
                        x = col.find(name)
                        if(x != -1): 
                            inner_array.append(1)
                            Found = True
                    if(not Found):
                        inner_array.append(0)
            #setup inputs Done
            features = np.array([inner_array])
            # using inputs to predict the output
            warnings.filterwarnings("ignore")
            prediction = lr_clf.predict(features)
            probabilities = lr_clf.predict_proba(X_test)
            odds = max(probabilities[0]) * 100
            if(odds == 100): odds = 99.99       #nothing is guaranteed
            if(prediction[0]==1):
                print("Prediction: The patient has Chronic Kidney Disease(CKD)")
            else:
                print("Prediction: The patient does not have Chronic Kidney Disease(CKD)")
            print("Confidence: {:.2f}%".format(odds))
        #Done inputs and Prediction
        print("")
    #Exit loop
    #more value for 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

if __name__ == '__main__':
    #initial data
    df = pd.read_csv("Chronic_Kidney_Disease.csv", sep=",", header=0)
    
    #Start program
    option = 0
    while(option != 3):
        option = select_menu()
        print("")
        if(option == 1):
            print("<<Exploratory Data Analysis>>")
            data_analysis(df)
            print("")
        elif(option == 2):
            print("<<Logistic Regression>>")
            data_processing(df)
    #exited with option = 3
    print("<You exited the Chronic Kidney Disease predictor program>")
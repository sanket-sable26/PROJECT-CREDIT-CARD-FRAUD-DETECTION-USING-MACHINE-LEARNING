def main():
    
    # Importing The libraries
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import os
    path = os.getcwd()
    
    #######################
    
    # Reading the dataset using Pandas 
    
    df = pd.read_csv('C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\creditcard.csv')
    
    ############
    ### Visualization
    
    # Graph 1 ---> Distribution Plots of Time and Amount
    
    fig, ax = plt.subplots(1, 2, figsize=(18,4))

    amount_val = df['Amount'].values
    time_val = df['Time'].values
    
    sns.distplot(amount_val, ax=ax[0], color='r')
    ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])
    
    sns.distplot(time_val, ax=ax[1], color='b')
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])



    plt.show()
    
    ########################
    ### Graph 2 ---> Boxplots with respect to target variable
    
    new_df = df.copy()
    colors = ["#0101DF", "#DF0101"]
    f, axes = plt.subplots(ncols=4, figsize=(20,4))

    # Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
    sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])
    axes[0].set_title('V11 vs Class Positive Correlation')
    
    sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
    axes[1].set_title('V4 vs Class Positive Correlation')
    
    
    sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])
    axes[2].set_title('V2 vs Class Positive Correlation')
    
    
    sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])
    axes[3].set_title('V19 vs Class Positive Correlation')
    
    plt.show()
    
    ########################
    ### Graph 3 ---> Continuous Distribution Graph 
    
    from scipy.stats import norm

    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))
    
    v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
    sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
    ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)
    
    v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
    sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
    ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)
    
    
    v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
    sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
    ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
    
    plt.show()
    
    # Model Building
    # Scaling Using Robust Scaler.
    # RobustScaler is less prone to outliers.
    
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()
    
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    X = df.drop('Class',axis = 1) # X is input
    y = df['Class'] # y is output
    
    # Machine Learning Modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = LogisticRegression(max_iter = 10000)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = accuracy_score(y_test, preds)
    b = "Accuracy: %.2f%%" % (acc*100)
    
    result = pd.DataFrame({'ID':X_test['ID'],'Actual Observations': y_test, 'Predicted Observations' : preds})
    result.to_csv('C:\\Users\\sunil\\Projects\\Credit Card Fraud Detection\\Result.csv', index=False)
    return b
main()
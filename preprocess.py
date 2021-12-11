import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.9
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

if __name__ == '__main__':
    file_dtypes = {'Ticket number': str, 'Issue time': np.float64, 'Meter Id': str, 
    'Marked Time': str, 'RP State Plate': str, 'Plate Expiry Date': str, 'VIN': str, 
    'Make': str, 'Body Style': str, 'Color': str, 'Location': str, 'Route': str, 
    'Agency': np.float64, 'Violation code': str, 'Violation Description': str, 
    'Fine amount': np.float32, 'Latitude': np.float32, 'Longitude': np.float32}

    df_tc = pd.read_csv('data/parking_citations.corrupted.csv', 
                        dtype=file_dtypes,
                        parse_dates=['Issue Date'],
                        index_col=['Ticket number'])

    df_tc = df_tc.drop(columns=['Meter Id', 'Marked Time', 'VIN'])
    print("imported successfully")

    df_tc_make_exists = df_tc.dropna(subset=['Make'])
    df_tc_make_exists = df_tc_make_exists.dropna()
    df_tc_make_exists = df_tc_make_exists.loc[df_tc_make_exists['Latitude'] != 99999]
    print("Make only and Lat=99999 removed")

    df_top25_makes = df_tc_make_exists.Make.value_counts().head(25)
    top25_arr = df_top25_makes.index.to_numpy()

    # df_tc_make_exists['Make_mod'] = np.where(df_tc_make_exists['Make'].isin(top25_arr), "TOP 25", "THE REST")
    df_tc_make_exists['Make_mod'] = np.where(df_tc_make_exists['Make'].isin(top25_arr), 1, 0)

    pd.set_option('display.max_columns', 30)
    print(df_tc_make_exists.head())

    df_tc_make_exists = df_tc_make_exists[['Issue time', 'Fine amount', 'Latitude', 'Longitude', 'Make_mod', 'RP State Plate', 'Body Style',
                                 'Color', 'Violation Description']]

    start_time_pdohe = time.time()
    features_pd = pd.get_dummies(data=df_tc_make_exists,
                                columns=['RP State Plate', 'Body Style',
                                 'Color', 'Violation Description'])
    pandas_ohe = time.time() - start_time_pdohe
    print(pandas_ohe)

    # df_tc_make_exists = pd.concat([df_tc_make_exists, features_pd], axis=1)

    labels = np.array(features_pd['Make_mod'])
    features_pd = features_pd.drop('Make_mod', axis=1)

    feature_list = list(features_pd.columns)

    features = np.array(features_pd)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    uniques, counts = np.unique(labels, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(labels)))
    print(percentages)

    uniques, counts = np.unique(test_labels, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(test_labels)))
    print(percentages)

    # start_train = time.time()
    # model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # model.fit(train_features, train_labels)
    # print("It took ", time.time() - start_train, " to train")

    # dump(model, 'rf_model.joblib')

    model = load('rf_model.joblib')

    train_rf_predictions = model.predict(train_features)
    train_rf_probs = model.predict_proba(train_features)[:, 1]

    rf_predictions = model.predict(test_features)
    rf_probs = model.predict_proba(test_features)[:, 1]

    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18

    evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
    plt.savefig('roc_auc_curve.png')

    # Confusion matrix
    cm = confusion_matrix(test_labels, rf_predictions)
    plot_confusion_matrix(cm, classes = ['THE REST', 'TOP 25'],
                        title = 'Confusion Matrix')

    plt.savefig('cm.png')
# import dependencies
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

### NOTE: All of this code is taken from course material in Data Mining (Amir Jafari) and Machine Learning 1 (Yuxiao Huang)

####################################################################################################
####################################################################################################
####################################################################################################
# MODEL A1
# spanning 1995-2015 without vhf location data
# using random forest for feature selection
####################################################################################################
####################################################################################################
####################################################################################################

# load model inputs data
data = pd.read_csv('Data/Model/Input/inputs.csv')

# trim invalid columns
data.drop(['Herd', 'Year', 'Status'], axis=1, inplace=True)

# initialize target variable and features to drop
target = 'StatusClass'
visitors = ['Visitors', 'VisEwes', 'VisRams', 'VisR', 'VisRT', 'VisT']

# drop ignored features
df = data.drop(visitors, axis=1)

# initialize features and labels
X = df.drop(target, axis=1)
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

# initialize decision tree pipeline and fit
pipe_dt = Pipeline([('StandardScaler', StandardScaler()), ('DecisionTreeClassifier', DecisionTreeClassifier())])
pipe_dt.fit(X, y)

dot_data = export_graphviz(pipe_dt.named_steps['DecisionTreeClassifier'], filled=True, rounded=True,
  class_names=['Healthy', 'Infected'], feature_names=feature_names, out_file=None)

# plot and save decision tree
graph = graph_from_dot_data(dot_data)
img = Image(graph.create_png())
open('Data/Model/Output/1995-2015-(NoVHF)/DT.png', 'wb').write(img.data)
img = Image(graph.create_svg())
open('Data/Model/Output/1995-2015-(NoVHF)/DT.svg', 'wb').write(img.data)

# initialize random forest pipeline and fit
pipe_rf = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestClassifier', RandomForestClassifier(n_estimators=100))])
pipe_rf.fit(X, y)

# curate, plot, and save random forest results
f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_, feature_names)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.savefig('Data/Model/Output/1995-2015-(NoVHF)/RF.png', bbox_inches='tight', pad_inches=0.25)
plt.show()
f_importances = pd.DataFrame(f_importances, columns=['score'])
f_importances.reset_index(inplace=True)
f_importances.columns.values[0] = 'feature'
f_importances.to_csv('Data/Model/Output/1995-2015-(NoVHF)/RF.csv', index=False)

# initialize selected features from random forest results
candidates = ['Lambs', 'PopAdults', 'Area', 'Translocation', 'NonResident']

# reset features and labels
X = df[candidates]
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

# TT split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# normalize the data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize and fit logistic regression
lr = LogisticRegression(C=1000.0, random_state=0, solver="liblinear")
lr.fit(X_train, y_train)

# predict, score, and plot results

y_pred = lr.predict(X_test)
y_pred_score = lr.predict_proba(X_test)

print("A1 accuracry:", str(accuracy_score(y_test, y_pred) * 100))
print("A1 ROC AUC:", str(roc_auc_score(y_test, y_pred_score[:,1]) * 100))

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['StatusClass'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
  yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('Data/Model/Output/1995-2015-(NoVHF)/CM(selected).png', bbox_inches='tight', pad_inches=0.25)

####################################################################################################
# MODEL A2
# spanning 1995-2015 without vhf location data
# using all features
####################################################################################################

# load model inputs data
data = pd.read_csv('Data/Model/Input/inputs.csv')

# trim invalid columns
data.drop(['Herd', 'Year', 'Status'], axis=1, inplace=True)

# initialize target variable and features to drop
target = 'StatusClass'
visitors = ['Visitors', 'VisEwes', 'VisRams', 'VisR', 'VisRT', 'VisT']

# drop ignored features
df = data.drop(visitors, axis=1)

# initialize features and labels
X = df.drop(target, axis=1)
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

# TT split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# normalize the data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize and fit logistic regression
lr = LogisticRegression(C=1000.0, random_state=0, solver="liblinear")
lr.fit(X_train, y_train)

# predict, score, and plot results

y_pred = lr.predict(X_test)
y_pred_score = lr.predict_proba(X_test)

print("A2 accuracy:", str(accuracy_score(y_test, y_pred) * 100))
print("A2 ROC AUC:", str(roc_auc_score(y_test, y_pred_score[:,1]) * 100))

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['StatusClass'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
  yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('Data/Model/Output/1995-2015-(NoVHF)/CM(full).png', bbox_inches='tight', pad_inches=0.25)
plt.close()

####################################################################################################
####################################################################################################
####################################################################################################
# MODEL B1
# spanning 1997-2012 with vhf data
# using random forest for feature selection
####################################################################################################
####################################################################################################
####################################################################################################

# load model inputs data
data = pd.read_csv('Data/Model/Input/inputs.csv')

# trim invalid columns
data.drop(['Herd', 'Year', 'Status'], axis=1, inplace=True)

# initialize target
target = 'StatusClass'

# drop ignored records
df = data.dropna(subset=['Visitors'])

# initialize features and labels
X = df.drop(target, axis=1)
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)
# initialize decision tree pipeline and fit
pipe_dt = Pipeline([('StandardScaler', StandardScaler()), ('DecisionTreeClassifier', DecisionTreeClassifier())])
pipe_dt.fit(X, y)

dot_data = export_graphviz(pipe_dt.named_steps['DecisionTreeClassifier'], filled=True, rounded=True,
  class_names=['Healthy', 'Infected'], feature_names=feature_names, out_file=None)

# plot and save decision tree
graph = graph_from_dot_data(dot_data)
img = Image(graph.create_png())
open('Data/Model/Output/1997-2012-(WithVHF)/DT.png', 'wb').write(img.data)
img = Image(graph.create_svg())
open('Data/Model/Output/1997-2012-(WithVHF)/DT.svg', 'wb').write(img.data)

# initialize random forest pipeline and fit
pipe_rf = Pipeline([('StandardScaler', StandardScaler()), ('RandomForestClassifier', RandomForestClassifier(n_estimators=100))])
pipe_rf.fit(X, y)

# curate, plot, and save random forest results
f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_, feature_names)
f_importances = f_importances.sort_values(ascending=False)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=14)
plt.savefig('Data/Model/Output/1997-2012-(WithVHF)/RF.png', bbox_inches='tight', pad_inches=0.25)
plt.show()
f_importances = pd.DataFrame(f_importances, columns=['score'])
f_importances.reset_index(inplace=True)
f_importances.columns.values[0] = 'feature'
f_importances.to_csv('Data/Model/Output/1997-2012-(WithVHF)/RF.csv', index=False)

# initialize selected features from random forest results
candidates = ['Lambs', 'PopAdults', 'Area', 'Translocation', 'VisRams']

# reset features and labels
X = df[candidates]
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

# TT fit the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# normalize the data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize and fit logistic regression
lr = LogisticRegression(C=1000.0, random_state=0, solver="liblinear")
lr.fit(X_train, y_train)

# predict, score, and plot results

y_pred = lr.predict(X_test)
y_pred_score = lr.predict_proba(X_test)

print("B1 accuracy:", str(accuracy_score(y_test, y_pred) * 100))
print("B2 ROC AUC:", str(roc_auc_score(y_test, y_pred_score[:,1]) * 100))

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['StatusClass'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
  yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('Data/Model/Output/1997-2012-(WithVHF)/CM(selected).png', bbox_inches='tight', pad_inches=0.25)

####################################################################################################
# MODEL B2
# spanning 1997-2012 with vhf data
# using all features
####################################################################################################

# load model inputs data
data = pd.read_csv('Data/Model/Input/inputs.csv')

# trim invalid columns
data.drop(['Herd', 'Year', 'Status'], axis=1, inplace=True)

# initialize target
target = 'StatusClass'

# drop ignored records
df = data.dropna(subset=['Visitors'])

# initialize features and labels
X = df.drop(target, axis=1)
y = df[target]

# store feature names for plots
feature_names = X.columns

# balance datasets
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

# TT split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# normalize the data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize and fit logistic regression
lr = LogisticRegression(C=1000.0, random_state=0, solver="liblinear")
lr.fit(X_train, y_train)

# predict, score, and plot results

y_pred = lr.predict(X_test)
y_pred_score = lr.predict_proba(X_test)

print("B2 accuracy:", str(accuracy_score(y_test, y_pred) * 100))
print("B2 ROC AUC:", str(roc_auc_score(y_test, y_pred_score[:,1]) * 100))

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['StatusClass'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
  yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('Data/Model/Output/1997-2012-(WithVHF)/CM(full).png', bbox_inches='tight', pad_inches=0.25)

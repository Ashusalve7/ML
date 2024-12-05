import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import io

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import preprocessing
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

data = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/refs/heads/master/candy-power-ranking/candy-data.csv')

data = data[['fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent','winpercent','chocolate']]
data.head()

training_set,test_set=train_test_split(data,test_size=0.2,random_state=2)

train_df = training_set.dropna()
test_df = test_set.dropna()

x_test,y_test = test_df.iloc[:,:-1].values,test_df.iloc[:,-1].values
x_train,y_train = train_df.iloc[:,:-1].values,train_df.iloc[:,-1].values

x_train = train_df[['fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent','winpercent','chocolate']]
y_train = train_df['chocolate']
x_test = test_df[['fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent','winpercent','chocolate']]
y_test = test_df['chocolate']

y_test.head()

y_train.value_counts()

sns.countplot(x ="chocolate",data=train_df,palette="Blues_d")
plt.savefig('chocolate.png')
plt.show()

count_no_choc = len(train_df[train_df['chocolate']==0])
count_choc = len(train_df[train_df['chocolate']==1])
pct_of_no_choc = count_no_choc/(count_no_choc+count_choc)
print("percentage of no chocolate: ",pct_of_no_choc*100)
pct_of_choc = count_choc/(count_no_choc+count_choc)
print("percentage of chocolate: ",pct_of_choc*100)

train_df.groupby('chocolate').mean()

train_df.groupby('caramel').mean()

from sklearn.linear_model import LogisticRegression

logR = LogisticRegression()
logR.fit(x_train,y_train)
ypred = logR.predict(x_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,ypred)
cnf_matrix

class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('confusion matrix',y=1.1)
plt.ylabel('actual label')
plt.xlabel('predicted label')
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test,ypred))
print("Precision:",metrics.precision_score(y_test,ypred))
print("Recall:",metrics.recall_score(y_test,ypred))

y_pred_proba = logR.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


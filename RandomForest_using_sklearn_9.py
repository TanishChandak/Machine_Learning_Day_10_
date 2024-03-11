import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

# printing the column elements which is present:
print(dir(digits))

# plotting the graph:
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
    # plt.show()

# Conveting the data into dataframe:
df = pd.DataFrame(digits.data)
print(df.head())

# adding the (target) column into the dataset:
df['target'] = digits.target
print(df.head())

#independent variables:
X = df.drop(['target'], axis='columns')
print(X.head())

# dependent variables:
y = df.target
print(y.head())

# training and testing the data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(len(X_test))
print(len(X_train))

# RandomForest Classifier:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)

# training the model:
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
y_predicted = model.predict(X_test)

# confusion metrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

# plotting the heapmap using the seabon:
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
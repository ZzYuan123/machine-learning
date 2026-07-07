from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X=[
    [0],
    [1],
    [2],
    [3],
    [5],
    [8]
]

y=[
    "正常",
    "正常",
    "正常",
    "垃圾",
    "垃圾",
    "垃圾"
]

model=LogisticRegression()

model.fit(X,y)

y_pred=model.predict(X)

print(confusion_matrix(y,y_pred))

print(classification_report(y,y_pred))
from sklearn.linear_model import LogisticRegression

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

print(model.predict([[0.5]]))
print(model.predict([[7]]))

print(model.predict_proba([[0.5]]))
print(model.predict_proba([[7]]))

print(model.classes_)
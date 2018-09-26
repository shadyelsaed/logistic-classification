# logistic-classification
Using logistic regression to train and predict whether Mushroom is poisonous or edible
# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

pred1 = reg.predict(X_test)
In [89]:
# trying to evaluate the model using confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
print('logistic regression accuracy')
print(confusion_matrix(pred1, y_test))
print('-------------------------------------')
print(classification_report(pred1, y_test))
logistic regression accuracy
[[831 107]
 [ 33 654]]
-------------------------------------
             precision    recall  f1-score   support

          0       0.96      0.89      0.92       938
          1       0.86      0.95      0.90       687

avg / total       0.92      0.91      0.91      1625

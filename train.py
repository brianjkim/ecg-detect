from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from load_data import get_train_labels
from features import features

labels = get_train_labels()
data = features

# simple machine learning using random forest classifier
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
test_predict = forest.predict(test_data)
print(classification_report(test_labels, test_predict, target_names=['N', 'S', 'V', 'F', 'Q']))

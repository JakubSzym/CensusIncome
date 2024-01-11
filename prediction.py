import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def find_best_k_for_knn(x_train, y_train, x_test, y_test, max_k):
    accuracy_scores = []

    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'K={k}, Accuracy={accuracy}')

    
    plt.figure(f)
    plt.plot(range(1, max_k + 1), accuracy_scores, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, max_k + 1))
    plt.savefig('knn_accuracy_vs_k.png')
    plt.show()

    best_k = accuracy_scores.index(max(accuracy_scores)) + 1
    return best_k, max(accuracy_scores)


df = pd.read_csv("lighter-data.csv")


label_encoder = LabelEncoder()
categorical_columns = ['income', 'workclass', 'occupation', 'relationship', 'race', 'sex','native-country']
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['income']), df['income'], test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#model = SVC()
#model = KNeighborsClassifier(n_neighbors=27)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('report: \n.', classification_report(y_test, y_pred), '\n')
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()
plt.savefig('confusion_matrix.png')



#best_k, best_accuracy = find_best_k_for_knn(x_train, y_train, x_test, y_test, 50)
#print(f'k: {best_k}, accuracy: {best_accuracy}')




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load a sample dataset
# data = sns.load_dataset('titanic')

# # Data manipulation
# # Filter passengers older than 30
# filtered_data = data[data['age'] > 30]

# # Group data by class and calculate mean age
# grouped_data = data.groupby('class')['age'].mean()

# # Display manipulated data
# print("Filtered Data (Age > 30):\n", filtered_data)
# print("\nAverage Age by Class:\n", grouped_data)

# # Visualization
# plt.figure(figsize=(10, 6))
# sns.countplot(data=data, x='class', hue='survived')
# plt.title('Survival Count by Class')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = sns.load_dataset('titanic')

data = data.dropna(subset=['age', 'fare', 'sex', 'class', 'survived'])

# Encoding categorical variables
le_sex = LabelEncoder()
le_class = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])
data['class'] = le_class.fit_transform(data['class'])

# Feature selection
X = data[['age', 'fare', 'sex', 'class']]
y = data['survived']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model and tracking accuracy
accuracy_list = []
final_model = None
y_final_pred = None
for i in range(1, 101):
    model = LogisticRegression(max_iter=i, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    if i == 100:
        final_model = model
        y_final_pred = y_pred

# Plot accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), accuracy_list, marker='o', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iterations for Logistic Regression')
plt.grid()
plt.show()

# Confusion Matrix
if final_model is not None and y_final_pred is not None:
    cm = confusion_matrix(y_test, y_final_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

# Display manipulated data
filtered_data = data[data['age'] > 30]
grouped_data = data.groupby('class')['age'].mean()
print("Filtered Data (Age > 30):\n", filtered_data)
print("\nAverage Age by Class:\n", grouped_data)

# Visualization of survival count
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='class', hue='survived')
plt.title('Survival Count by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

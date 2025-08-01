import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].mean(), inplace=True)

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\nMissing values after handling:\n",df.isnull().sum())

df['Sex']=LabelEncoder().fit_transform(df['Sex'])
df['Embarked']=LabelEncoder().fit_transform(df['Embarked'])

numeric_df=df.select_dtypes(include=['float64','int64'])
numeric_df=numeric_df.fillna(numeric_df.mean())

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Feature correlation heatmap")
plt.show()

X=df.drop("Survived", axis=1)
y=df["Survived"]

features=['Age','Fare','Pclass','SibSp','Parch']

X=df[features]
X=X.fillna(X.median())

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled_df=pd.DataFrame(X_scaled, columns=features)
X_scaled_df.head()
x_train,x_test,y_train,y_test=train_test_split(X_scaled_df,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
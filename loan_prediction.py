import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("loan-train.csv")
df.dropna(subset=['Loan_Status'], inplace=True)

cols_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

df.drop("Loan_ID", axis=1, inplace=True)

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

y = df_encoded['Loan_Status_Y']
X = df_encoded.drop('Loan_Status_Y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train[cols_to_scale])
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n--- Model Performance ---")
print(f"Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
print("\ndetailed report:")
print(classification_report(y_test, y_pred))

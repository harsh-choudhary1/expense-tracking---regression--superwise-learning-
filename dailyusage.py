import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#  expense data
data = {
    "date": ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"],
    "category": ["food", "traveling", "entertainment", "fast_food", "car"],
    "amount": [200, 1000 , 150, 20, 1100]
}
df = pd.DataFrame(data)
print(df)

category_totals = df.groupby("category")["amount"].sum()
category_totals.plot(kind="pie", autopct='%1.1f%%', figsize=(6, 6))
plt.title("spending by category")
plt.show()

# future expenses
expense_data = {
    "month": [1, 2, 3, 4, 5],
    "total_expense": [1350, 1400, 1300, 1450, 1500]
}
expense_df = pd.DataFrame(expense_data)

X = expense_df[["month"]]
y = expense_df["total_expense"]

# linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# predicting next month's expenses
next_month = [[6]]  # predicting for month 6
predicted_expense = model.predict(next_month)
print(f"predicted expense for 6 month : Rs.{predicted_expense[0]:.2f}")

# plot expense trend
plt.plot(expense_df["month"], expense_df["total_expense"], marker='o', label="actual")
plt.plot(6, predicted_expense, 'ro', label="predicted")
plt.xlabel("month")
plt.ylabel("total xpense")
plt.legend()
plt.title("expense trend and prediction")
plt.show()

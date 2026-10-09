
# ==========================================
# STUDENT PASS PREDICTION - EDA
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the historical dataset
df = pd.read_csv("student_data.csv")

# 2. Display the first five rows
print("\nFIRST FIVE STUDENTS")
print(df.head())

# 3. Display dataset size
print("\nDATASET SHAPE")
print(df.shape)

# 4. Check column names and data types
print("\nDATASET INFORMATION")
df.info()

# 5. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 6. Check duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# 7. Display statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 8. Count pass and fail students
print("\nRESULT COUNTS")
print(df["Final_Result"].value_counts())

# 9. Plot pass and fail counts
sns.countplot(data=df, x="Final_Result")
plt.title("Number of Students Who Pass and Fail")
plt.show()

# 10. Explore study hours by result
sns.boxplot(data=df, x="Final_Result", y="Study_Hours")
plt.title("Study Hours vs Final Result")
plt.show()

# 11. Explore attendance by result
sns.boxplot(data=df, x="Final_Result", y="Attendance")
plt.title("Attendance vs Final Result")
plt.show()

# 12. Correlation between numerical features
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="Greens"
)
plt.title("Feature Correlation Heatmap")
plt.show()
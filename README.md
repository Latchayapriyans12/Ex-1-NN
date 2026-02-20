<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# Import Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ===============================
# Dataset
# ===============================
df = pd.read_csv("Churn_Modelling.csv")

print("\n\n==============================")
print("**Dataset**")
print("==============================\n")
display(df)

# ===============================
# X and Y Values
# ===============================
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

print("\n\n==============================")
print("**X Values**")
print("==============================\n")
display(X)

print("\n\n==============================")
print("**Y Values**")
print("==============================\n")
display(Y.to_frame())

# ===============================
# Null Values
# ===============================
print("\n\n==============================")
print("**Null Values**")
print("==============================\n")
display(df.isnull().sum().to_frame(name="Null Count"))

# ===============================
# Duplicated Values
# ===============================
print("\n\n==============================")
print("**Duplicated Values**")
print("==============================\n")
display(df.duplicated().to_frame(name="Duplicate"))

# ===============================
# Description
# ===============================
print("\n\n==============================")
print("**Description**")
print("==============================\n")
display(df.describe())

# ===============================
# Normalize Dataset
# ===============================
scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['int64','float64']).columns

normalized_df = df.copy()
normalized_df[num_cols] = scaler.fit_transform(df[num_cols])

print("\n\n==============================")
print("**Normalized Dataset**")
print("==============================\n")
display(normalized_df)

# ===============================
# Train Test Split
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("\n\n==============================")
print("**Training Data**")
print("==============================\n")
display(X_train)

print("\n\n==============================")
print("**Testing Data**")
print("==============================\n")
display(X_test)

# ===============================
# Length of Test Data
# ===============================
print("\n\n==============================")
print("**Length of Test Data**")
print("==============================\n")
print(len(X_test))

```

## OUTPUT:

<img width="1591" height="1199" alt="image" src="https://github.com/user-attachments/assets/bdb7cf34-3245-42a9-94ed-b2b895e0a93c" />

<img width="325" height="612" alt="image" src="https://github.com/user-attachments/assets/cb33b498-64c4-49e7-b660-d3f89fe6746a" />

<img width="378" height="691" alt="image" src="https://github.com/user-attachments/assets/7928991b-326f-41d4-806c-3bcff1decc25" />

<img width="389" height="599" alt="image" src="https://github.com/user-attachments/assets/db8a7a5b-e9c3-46b4-b294-bf3d600d176c" />

<img width="1641" height="475" alt="image" src="https://github.com/user-attachments/assets/5778fd4e-c21c-4b0b-b33f-67899769fa58" />

<img width="1590" height="622" alt="image" src="https://github.com/user-attachments/assets/7f1e6fd4-2b03-41dd-9d1f-294f74770252" />

<img width="1499" height="629" alt="image" src="https://github.com/user-attachments/assets/79ccb691-20de-453d-9795-738c88726993" />

<img width="1500" height="772" alt="image" src="https://github.com/user-attachments/assets/8edd8fb0-a543-43fc-9a72-dfd9b50c854e" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



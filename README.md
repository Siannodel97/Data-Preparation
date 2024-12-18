# Data Set Preparation

In this Notebook you'll learn how to prepare a basic data set where there are null values, categoric values, data scaling and normalization. This preparation is basic and essential for our model's training to get good predictions.

# Set up


```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler

# Imagine we have loaded our data set as "train_set", a DataFrame made by Pandas from our data file in .csv format.

file_path = "...\folder\data_set.csv"
train_set = pd.read_csv(file_path)

```

## Cleaning data

First of all, we are going tos separate the label from the rest of the set because we don't need to apply the same transformations in both sets.


```python
X_train = train_set.drop("label", axis=1) # Change "label" for the name of the label column
y_train = train_set["label"].copy()
```

Imagine our data set has null values in some features which Machine Learning algorithms can't work on. That's why exists many options to replace them like:
* Delete corresponding rows.
* Delete the corresponding feature (column).
* Replace the null value with a determined value (zero, mean, ...).

Let's check if exist any feature with null values:


```python
X_train.isna().any() # With .isna() you can check if there are null values.
```


```python
# Now we select the rows which contains null values in
null_values_rows = X_train[X_train.isnull().any(axis=1)]

# You can see them
null_values_rows
```

### Option 1: Delete corresponding rows with null values


```python
# Copy the training to not change the original
X_train_copy = X_train.copy()

# Once time you know which rows has null values you can delete them
X_train_copy.dropna(subset=["feature1","feature2","..."], inplace=True)
# Change the features name for those who have null values
```

We can count the number of deleted rows.


```python
print("Number of deleted rows:",len(X_train)-len(X_train_copy))
```

### Option 2: Delete features with null values


```python
# Copy the training to not change the original
X_train_copy = X_train.copy()

# Once time you know which features has null values you can delete them
X_train_copy.drop(["feature1","feature2","..."], axis=1, inplace=True)
# Change the features name for those who have null values
```

We can count the number of deleted features (columns).


```python
print("Number of deleted features:", len(list(X_train))-len(list(X_train_copy)))
```

### Option 3: Replace the null values with a determined value

#### Mean value


```python
# Copy the training to not change the original
X_train_copy = X_train.copy()

# Now we replace the null values with the mean of the feature's values
mean_feature1 = X_train_copy["feature1"].mean()
mean_feature2 = X_train_copy["feature2"].mean()

X_train_copy["feature1"] = X_train_copy["feature1"].fillna(mean_feature1)
X_train_copy["feature2"] = X_train_copy["feature2"].fillna(mean_feature2)
# Change the features name for those who have null values
```

#### Median value


```python
# Copy the training to not change the original
X_train_copy = X_train.copy()

# Now we replace the null values with the median of the feature's values
median_feature1 = X_train_copy["feature1"].median()
median_feature2 = X_train_copy["feature2"].median()

X_train_copy["feature1"] = X_train_copy["feature1"].fillna(median_feature1)
X_train_copy["feature2"] = X_train_copy["feature2"].fillna(median_feature2)
# Change the features name for those who have null values
```

### Option 3 alternative: Sklearn - SimpleImputer class


```python
# Copy the training to not change the original
X_train_copy = X_train.copy()

imputer = SimpleImputer(strategy="median")

# Imputer class doesn't allow categoric values so we need to delete them
X_train_copy_num = X_train_copy.select_dtypes(exclude=["object"])
# Check that the subset only have numeric values
X_train_copy_num.info()
```


```python
# Now we proporcionate the numeric values to calculate the median
imputer.fit(X_train_copy_num)
```


```python
# Replace the null values
X_train_copy_num_nonan = imputer.transform(X_train_copy_num)
```


```python
# Transform the result to a DataFrame from Pandas
X_train_copy = pd.DataFrame(X_train_copy_num_nonan, columns=X_train_copy_num.columns)
```

## Transform categoric features to numeric features

Before start with the transformation, let's bring the clean data set and split the features and the label in two subsets.


```python
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()
```

Imagine our data set has plenty of categoric values so we need to transform them into numeric values.


```python
# Let's see which features has categoric values (dtype = object)
X_train.info()
```

There are many ways to transform categoric features into numeric values. One of them (and probably the simplest) is using the **factorize** method from Pandas which transform each categoric into sequential number.


```python
feature1 = X_train["feature1"]
feature1_encoded, categories = feature1.factorize()

# Let's check how they has been encoded
for i in range(10):
    print(feature1.iloc[i], "=", feature1_encoded.iloc[i])
```

#### Advanced transformations with sklearn

##### Ordinal Encoding

It does the same codification as **factorize** from Pandas.


```python
feature1 = X_train[['feature1']]

ordinal_encoder = OrdinalEncoder()
feature1_encoded = ordinal_encoder.fit_transform(feature1)
```

This type of codification has one problem. Certains ML algorithms which run measuring the similarity between two points by distance will consider that 1 is closer than 3 from 2. It doesn't make sense in categorical values. So it is not used for these cases.

##### One-Hot Encoding

Generate for each category from categoric feature one binary matrix which represent the value.


```python
# The sparse matrix only get the positions of the values which aren't '0' to save memory
feature1 = X_train[['feature1']]

oh_encoder = OneHotEncoder()
feature1_oh = oh_encoder.fit_transform(feature1)
feature1
```


```python
# Convert the sparse matrix to Numpy array
feature1_type_oh.toarray()
```


```python
# We can see how it has been encoded
for i in range(10):
    print(feature1["feature1"].iloc[i],"=",feature1_type_oh.toarray()[i])
print(ordinal_enconder.categories_)
```

In many cases, when partitioning the dataset or making a prediction with new examples, new values for certain categories may appear, causing an error in the **transform() function**. The **OneHotEncoding** class provides the **handle_unknown** parameter to either raise an error or ignore unknown categorical features during the transformation (the default behavior is to raise an error).

When this parameter is set to "ignore," and an unknown category is encountered during the transformation, the resulting encoded columns for that feature will contain only zeros. In inverse transformation, an unknown category will be denoted as None.


```python
oh_encoder = OneHotEncoder(handle_unknown='ignore')
```

##### Get Dummies

Get Dummies is a simple method which allows to apply One-Hot Encoding to a DataFrame from Pandas.


```python
pd.get_dummies(X_train["feature1"])
```

## Scale the data set

As same as the previous section, before commence with the scaling let's bring the clean data set and split the features and the label in two subsets.


```python
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()
```

In general, Machine Learning algorithms doesn't have good predictions if the values of the input features has very different ranges. Because of that, different scaling techniques are used. It is important to note that these scaling mechanisms should not be applied to the labels.

#### **Scaling Techniques**
* **StandardScaler:** Scales data to have a mean of 0 and a standard deviation of 1.
* **MinMaxScaler:** Scales data to a fixed range (default: 0 to 1).
* **MaxAbsScaler:** Scales data by dividing by the maximum absolute value of each feature. Keeps data in the range [-1,1].
* **RobustScaler:** Scales data using the median and interquartile range. Effective for datasets with outliers.

#### **Normalization Techniques**
* **Normalizer:** Re-scales each row (sample) to have a unit norm. Commonly used for text classification or clustering tasks.

**It's important to do the transformations only on the train set for try these values. Later, it will be applied on the test set.**


```python
# An example of scaling with RobustScaler
scale_attrs = X_train[["feature1","feature2","..."]] # Change the names of features

robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(scale_attrs)

X_train_scaled = pd.DataFrame(X_train_scaled, colums=["feature1","feature2","..."])
```

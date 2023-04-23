import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

#reading dataset

df = pd.read_csv("train.csv", header=0, parse_dates=True)
df.head()

#Shape of data

print('No. of Examples : ',df.shape[0])
print('No. of Features : ', df.shape[1])

# Attribute information

df.info()

# checking missing values

df.isnull().sum()

# EDA & Data Preprocessing

# vendor_id - a code indicating the provider associated with the trip record

sns.set_context('talk')

plt.figure(figsize=(8, 8))
sns.countplot(df['vendor_id'], palette='Dark2')
plt.title("Vendor ID")

# After analyzing the visualization above, we can say that there are service providers.
# 2nd Service provider is the most opted one by New Yorkers.

# Store & Forward flag

plt.figure(figsize=(8, 8))
plt.pie(df['store_and_fwd_flag'].value_counts(), colors=['lightgreen', 'lightcoral'], shadow=True, explode=[0.5,0], autopct='%1.2f%%', startangle=200)
plt.legend(labels=['Y', 'N'])
plt.title("Store and Forward Flag")

# store_and_fwd_flag: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server
# Y: store and forward; N: not a store and forward trip.

# Visualization tells us that there were very few trips of which the records were stored in memory due to no connection to the server.

# Label Encoding Features having Categorical Values

enc = LabelEncoder()
df['store_and_fwd_flag'] = enc.fit_transform(df['store_and_fwd_flag'])
df['vendor_id'] = enc.fit_transform(df['vendor_id'])

# Conversion of 'store_and_fwd_flag' and 'vendor_id' to be Label encoded as those are Categorical features , binarizing them will help us to compute them with ease.

# Descriptive Statistics

# Descriptive Stats

plt.figure(figsize=(25, 10))
sns.heatmap(df.describe()[1:], annot=True, cmap='gist_earth')
plt.title('Descriptive Stats')

# We can observe that there were trips having 0 passengers which we can consider as false trip.

# Also, there are trips having trip duration upto 3526282 seconds (Approx. 980 hours) which is kind of impossible in a day.

# Visualising Trip duration we can clearly notice few outliers at extreme right

plt.figure(figsize=(20, 5))
sns.boxplot(df['trip_duration'])

# Probably in this visualization we can clearly see some outliers , their trips are lasting between 1900000 seconds (528 Hours) to somewhere around 3500000 (972 hours) seconds which is impossible in case of taxi trips, How can a taxi trip be that long?

# It’s Quite suspicious. We’ll have to get rid of those Outliers.

# Spread of Passenger count

plt.figure(figsize=(20, 5))
sns.boxplot(df['passenger_count'], color='maroon')
plt.title('Passenger Count Distribution')

# Most number of trips are done by 1-2 passenger(s).

# But one thing is Interesting to observe, there exist trip with Zero passengers, was that a free ride ? Or just a False data recorded ?

# Above 4 Passengers Indicate that the cab must be larger sized.

# Log Transformation

plt.figure(figsize=(10, 8))
sns.distplot(np.log(df['trip_duration']), kde=False, color='black')
plt.title("Log Transformed - Trip Duration")

# Since our Evaluation Metric is RMSLE, we'll proceed further with Log Transformed "Trip duration".

# Log Transformation Smoothens outliers by proving them less weightage.

# Passenger count

plt.figure(figsize=(10, 7))
sns.countplot(df['passenger_count'], palette='pastel')

# Above visualization tells us that there were most number of trips are done by Single passenger.

# 5 - 9 passengers trip states us that cab must be a Large vehicle.

# Feature Engineering

# Extracting day, month, date, hour, mins, weekday from datetime

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_date'] = df['pickup_datetime'].dt.date
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_min'] = df['pickup_datetime'].dt.minute
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

df['dropoff_min'] = df['dropoff_datetime'].dt.minute

# monthly trips exploration

plt.figure(figsize=(10, 7))
sns.countplot(df['pickup_month'], palette='Accent')
plt.xticks([0, 1, 2, 3, 4, 5], labels=['Jan', 'Feb', 'March', 'April', 'May', 'June'], rotation=90)
plt.title('Overall Monthly trips')

# Analyzing hourly pickups

plt.figure(figsize=(20, 5))
pickup_hour = df['pickup_hour'].value_counts()
pickup_hour.sort_index().plot(kind='bar', color='gold')
plt.title("Hourly Pickups")

# We get to see maximum pickups in rush hours (5 pm to 10 pm), probably office leaving time.

# Analyzing week day trips

plt.figure(figsize=(10, 7))
pickup_week = df['pickup_weekday'].value_counts()
pickup_week.sort_index().plot(kind='bar', color='maroon')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=90)
plt.title('Overall Week day Trips')

# Observations tells us that Fridays and Saturdays are those days in a week when New Yorkers prefer to rome in the city.

# Examining Daily trip

plt.figure(figsize=(20, 15))
df['pickup_day'].value_counts().plot(color="black", marker="+")
plt.title('Daily Trips Plot')

# Seem like New Yorker’s do not prefer to get a Taxi on Month end’s, there is a significant drop in the Taxi trip count as month end’s approach.

# Correlation Heatmap

# Plotting Pearson Correlation heatmap

plt.figure(figsize=(20, 10))
sns.heatmap(df.corr()*100, annot=True, cmap='inferno')
plt.title('Correlation Plot')

# Dropping unwanted columns

nyc_taxi_df = df.drop(['id', 'pickup_datetime', 'pickup_date', 'dropoff_datetime'], axis=1)
nyc_taxi_df.head()

# Let’s drop unwanted Features like ID and others of which we've already extracted features.

# Normalization

# Predictors and Target Variable

X = nyc_taxi_df.drop(['trip_duration'], axis=1)
y = np.log(nyc_taxi_df['trip_duration'])

# Normalising Predictors and creating new dataframe

cols = X.columns

ss = StandardScaler()

new_df = ss.fit_transform(X)
new_df = pd.DataFrame(new_df, columns=cols)
new_df.head()

# Normalizing the Dataset using Standard Scaling Technique.

X = new_df

# Applying PCA

pca = PCA(n_components=len(nyc_taxi_df.columns)-1)
pca.fit_transform(X)
var_rat = pca.explained_variance_ratio_
var_rat

# Variance Ratio vs PC plot

plt.figure(figsize=(15,6))
plt.bar(np.arange(pca.n_components_), pca.explained_variance_, color="grey")

# Cumulative Variance Ratio

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(var_rat)*100, color="g", marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.title('Elbow Plot')

# Applying PCA as per required components

pca = PCA(n_components=12)
transform = pca.fit_transform(X)
pca.explained_variance_

plt.figure(figsize=(25,6))
sns.heatmap(pca.components_, annot=True, cmap="winter")
plt.ylabel("Components")
plt.xlabel("Features")
plt.xticks(np.arange(len(X.columns)), X.columns, rotation=65)
plt.title('Contribution of a Particular feature to our Principal Components')

# Splitting Data and Choosing Algorithms

# Passing in Transformed values as Predictors

X = transform
y = np.log(nyc_taxi_df['trip_duration']).values

# RMSLE as a evaluation Metrics , We can also hyper tune our Parameters to minimize the loss (RMSLE). We will also calculate Null RMSLE , which we can set as a benchmark for our Model's RMSLE.

# importing train test split & some important metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Linear Regression

# Implementing Linear regression

est_lr = LinearRegression()
est_lr.fit(X_train, y_train)
lr_pred = est_lr.predict(X_test)
lr_pred

# coeficients & intercept

est_lr.intercept_, est_lr.coef_

# examining scores

print("Training Score : ", est_lr.score(X_train, y_train))

print("Validation Score : ", est_lr.score(X_test, y_test))

print("Cross Validation Score : ", cross_val_score(est_lr, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(lr_pred, y_test))

# prediction vs real data

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(lr_pred, kde=False, color="g", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# From the above visualization we can clearly identify that the Linear Regression isn't performing good.
# The Actual Data (in Grey) and Predicted values (in Yellow) are so much differing. We can conclude that Linear Regression doesn't seem like a right choice for Trip duration prediction.

# null rmsle implementation

y_null = np.zeros_like(y_test, dtype=float)
y_null.fill(y_test.mean())
print("Null RMSLE : ", np.sqrt(mean_squared_log_error(y_test, y_null)))

# Decision Tree

# Implementation of decision tree

est_dt = DecisionTreeRegressor(criterion="mse", max_depth=10)
est_dt.fit(X_train, y_train)
dt_pred = est_dt.predict(X_test)
dt_pred

# Examining metrics

print("Training Score : ", est_dt.score(X_train, y_train))

print("Validation Score : ", est_dt.score(X_test, y_test))

print("Cross Validation Score : " , cross_val_score(est_dt, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(dt_pred, y_test))

print("RMSLE : ", np.sqrt(mean_squared_log_error(dt_pred, y_test)))

# Prediction vs real data

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(dt_pred, kde=False, color="cyan", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# From the above we can clearly identify that the Decision Tree Algorithm is performing good. The Actual Data (in Grey) and Predicted values (in Red) are as close as possible. We can conclude that Decision Tree could be a good choice for Trip duration prediction.

# Parameter Tuning Decision Tree

# We can perform some hyper tuning on our Algorithm to get the most out of it, Hyper Tuning might consume lot of time and resources of the system depending upon the how big the Data we have and what algorithm we're using. It will go through number of Iterations and try to come up with the best possible value for us.

# Hyper parameter tuning

'''params = {'max_depth':[10,11,12,None], "min_samples_split":[2,3,4,5], 'max_features':[2,5,7,10]}
grid = GridSearchCV(est_dt, params, cv=5)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)
print (grid_pred)
grid.best_params_'''

# Random Forest

# random forest implementation

est_rf = RandomForestRegressor(criterion="mse", n_estimators=5, max_depth=10)
est_rf.fit(X_train, y_train)
rf_pred = est_rf.predict(X_test)
rf_pred

#examining metrics

print("Training Score : ", est_rf.score(X_train, y_train))

print("Validation Score : ", est_rf.score(X_test, y_test))

print("Cross Validation Score : ", cross_val_score(est_rf, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(rf_pred, y_test))

print("RMSLE : ", np.sqrt(mean_squared_log_error(rf_pred, y_test)))

# prediction vs real data

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(rf_pred, kde=False, color="indigo", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# r2 score plot for all 3 models

plt.figure(figsize=(10, 7))
r2 = pd.DataFrame({'Scores':np.array([r2_score(lr_pred, y_test), r2_score(dt_pred, y_test), r2_score(rf_pred, y_test)]), 'Model':np.array(['Linear Regression', 'Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="brown")
plt.axhline(y=0, color='g')
plt.title("R2 Scores")

# RMSLE Evaluation

# RMSLE plot

plt.figure(figsize=(10, 10))
r2 = pd.DataFrame({'RMSLE': np.array([np.sqrt(mean_squared_log_error(dt_pred, y_test)), np.sqrt(mean_squared_log_error(rf_pred, y_test))]), 'Model':np.array(['Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="lightblue", legend=False)
plt.title("RMSLE - Lesser is Better")

# We can see that our Decision Tree model and Random Forest model are good performers. As, Random Forest is providing us reduced RMSLE, we can say that it's a model to Opt for.

# The Second Approach - Without PCA

X = new_df
y = np.log(nyc_taxi_df['trip_duration']).values

# Linear Regression

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

# Implenting linear regression

est_lr = LinearRegression()
est_lr.fit(X_train, y_train)
lr_pred = est_lr.predict(X_test)
lr_pred

# Intercept & Coef

est_lr.intercept_, est_lr.coef_

# Examining metrics

print("Training Score : ", est_lr.score(X_train, y_train))

print("Validation Score : ", est_lr.score(X_test, y_test))

print("Cross Validation Score : ", cross_val_score(est_lr, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(lr_pred, y_test))

# prediction vs validation data

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(lr_pred, kde=False, color="g", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# Decision Tree

# Decision tree implementation

est_dt = DecisionTreeRegressor(criterion="mse", max_depth=10)
est_dt.fit(X_train, y_train)
dt_pred = est_dt.predict(X_test)
dt_pred

# examining metrics

print("Training Score : ", est_dt.score(X_train, y_train))

print("Validation Score : ", est_dt.score(X_test, y_test))

print("Cross Validation Score : ", cross_val_score(est_dt, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(dt_pred, y_test))

print("RMSLE : ", np.sqrt(mean_squared_log_error(dt_pred, y_test)))

# This time our decision tree model is not trained well as we can identify from scores.

# Prediction vs reality check

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(dt_pred, kde=False, color="cyan", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# Random Forest

# implementation of forest algorithm

est_rf = RandomForestRegressor(criterion="mse", n_estimators=5, max_depth=10)
est_rf.fit(X_train, y_train)
rf_pred = est_rf.predict(X_test)
rf_pred

# Examining metrics

print("Training Score : ", est_rf.score(X_train, y_train))

print("Validation Score : ", est_rf.score(X_test, y_test))

print("Cross Validation Score : ", cross_val_score(est_rf, X_train, y_train, cv=5).mean())

print("R2_Score : ", r2_score(rf_pred, y_test))

print("RMSLE : ", np.sqrt(mean_squared_log_error(rf_pred, y_test)))

# prediction vs reality check

plt.figure(figsize=(15, 8))
plt.subplot(1, 1, 1)
sns.distplot(y_test, kde=False, color="black", label="Test")

plt.subplot(1, 1, 1)
sns.distplot(rf_pred, kde=False, color="indigo", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")

# r2 score plot for all 3 models

plt.figure(figsize=(8, 7))
r2 = pd.DataFrame({'Scores':np.array([r2_score(lr_pred, y_test), r2_score(dt_pred, y_test), r2_score(rf_pred, y_test)]), 'Model':np.array(['Linear Regression', 'Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="maroon")
plt.axhline(y=0, color='g')
plt.title("R2 Scores")

# RMSLE plot

plt.figure(figsize=(8, 7))
r2 = pd.DataFrame({'RMSLE': np.array([np.sqrt(mean_squared_log_error(dt_pred, y_test)), np.sqrt(mean_squared_log_error(rf_pred, y_test))]), 'Model':np.array(['Decison Tree', 'Random Forest'])})
r2.set_index('Model').plot(kind="bar", color="skyblue", legend=False)
plt.title("RMSLE - Lesser is Better")

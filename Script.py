import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso
from sklearn.tree import DecisionTreeRegressor as dtReg
from sklearn import tree
from sklearn.svm import SVR
import xgboost as xgb
from pygam.pygam import LinearGAM, s, f
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

###############################################
#######                                 #######
#######    Data cleaning/engineering    #######
#######                                 #######
###############################################

# Import data
df = pd.read_csv(
    '/home/mich_qiu/PycharmProjects/DSML/PMA/PMA_blockbuster_movies.csv')

df.isna().sum() # check the number of NaNs in each feature
df.drop(['poster_url', '2015_inflation', 'genres', 'Genre_2', 'Genre_3',
         'studio', 'title', 'worldwide_gross', 'year'], axis=1, 
         inplace=True) # Drop irrelevant features
df.dropna(inplace=True) # Drop all NaNs

# Transform 'adjusted' data to numerical values
df['adjusted'] = [receipts[1:] for receipts in df['adjusted']]
df['adjusted'] = [float(receipts.replace(',', ''))
                  for receipts in df['adjusted']]

# Transform release date into months
df['release_date'] = [date[3:6] for date in df['release_date']]

# Create dummy features and concat to dataset
df_dummies = pd.get_dummies(df[['Genre_1', 'rating', 'release_date']])
df = pd.concat([df, df_dummies], axis=1)
df.drop(['Genre_1', 'rating', 'release_date'], axis=1, inplace=True) # drop non-encoded features


###############################################
#######                                 #######
#######    Exploratory data analysis    #######
#######                                 #######
###############################################

# Correlation matrix
correlation_matrix = df.corr().round(1)
sns.heatmap(data=correlation_matrix, annot=True)
df.drop(['imdb_rating'], axis=1, inplace=True) # remove 'imdb_rating'

# Find the movie counts in each category
counts = {}
df_counts = df.loc[:, 'Genre_1_Action':]
for i in df_counts.columns:
    counts[i] = df_counts[i].sum()
for i in counts:
    print('{}: {}'.format(i, counts[i]))

# Scatter plots for categorical features
plot_features = ['rt_audience_score', 'rt_freshness', 'length',
                 'Genre_1_Comedy', 'Genre_1_Fantasy', 'Genre_1_Thriller',
                 'Genre_1_History', 'Genre_1_Music',
                 'release_date_May', 'release_date_Jun', 'release_date_Jul',
                 'release_date_Dec', 'release_date_Feb', 'release_date_Sep']
for fea in plot_features:
    sns.scatterplot(x=fea, y='adjusted', data=df)
    plt.show(block=False)

# Standardise data
stdsc = StandardScaler()
df = pd.DataFrame(stdsc.fit_transform(df.values),
                  columns=df.columns, index=df.index)

# Train test split
df_target = df['adjusted']
df.drop(['adjusted'], axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(df, df_target, test_size=0.2, random_state=5)


###############################################
#######                                 #######
#######         Model selection         #######
#######                                 #######
###############################################

# Cross-Validation grid search
def crossValid(model, hyp, scores):
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        print("\n")
        clf = GridSearchCV(model, hyp, cv=5,
                           scoring=score)
        clf.fit(X_train, Y_train)
        print("Best parameters set found on the training set:")
        print(clf.best_params_)
        print("\n")


# Linear regression
linreg = LinearRegression()
linreg_fit = linreg.fit(X_train, Y_train)

# Training
linreg_pred = linreg_fit.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, linreg_pred)))
r2 = r2_score(Y_train, linreg_pred)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(r2))

# Test
linreg_pred = linreg_fit.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, linreg_pred)))
r2 = r2_score(Y_test, linreg_pred)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(r2))

# Lasso regression
lasso = LassoCV()
lasso.fit(X_train, Y_train)

# Training
predicted_lasso = lasso.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_lasso)))
train_score = lasso.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_lasso = lasso.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_lasso)))
test_score = lasso.score(X_test, Y_test)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Optimum alpha parameter
alpha = lasso.alpha_
print(alpha)
print('\n')


# Ridge regression
ridge = RidgeCV()
ridge.fit(X_train, Y_train)

# Training
predicted_ridge = ridge.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_ridge)))
train_score = ridge.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_ridge = ridge.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_ridge)))
test_score = ridge.score(X_test, Y_test)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Optimum alpha parameter
alpha = ridge.alpha_
print(alpha)
print('\n')


# Decision tree regression 
# (Hyperparameter optimised, please remove all parameters apart from random state for base model)
dtr = dtReg(random_state=23, criterion='mse', max_depth=None, max_features='sqrt',
            min_samples_split=11)
dtr_reg = dtr.fit(X_train, Y_train)

# Training
predicted_dtr = dtr.predict(X_train)
train_score = dtr.score(X_train, Y_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_dtr)))
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_dtr = dtr.predict(X_test)
test_score = dtr.score(X_test, Y_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_dtr)))
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Tree plot
tree.plot_tree(dtr)

# Cross-validation
tuned_parameters = [{'criterion': ['mse', 'friedman_mse', 'mae'],
                     'max_depth': [None, 3, 5, 7],
                     'min_samples_split': [3, 5, 7, 9, 11],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['neg_root_mean_squared_error', 'r2']
crossValid(dtr, tuned_parameters, scores)


# Support vector regression
# (Hyperparameter optimised, please remove all parameters for base model)
svr = SVR(C=10, gamma=0.001)
svr_fit = svr.fit(X_train, Y_train)

# Training
predicted_svr = svr.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_svr)))
train_score = svr.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_svr = svr.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_svr)))
test_score = svr.score(X_test, Y_test)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Cross-Validation
tuned_parameters = [{'kernel': ['sigmoid', 'rbf', 'poly', 'linear'], 'gamma': ['scale', 1e-3, 1e-4],
                     'C': [1, 0.5, 1.5, 10, 100]}]

scores = ['neg_root_mean_squared_error', 'r2']
crossValid(svr, tuned_parameters, scores)


# XGBoost regression
# (Hyperparameter optimised, please remove all parameters for base model)
xgb_clf = xgb.XGBRegressor(n_jobs=6, objective='reg:squarederror', booster='dart', training=True,
                           colsample_bytree=0.5, learning_rate=0.25, max_depth=5, n_estimators=25)
xgb_clf.fit(X_train, Y_train)

# Training
predicted_xgb = xgb_clf.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_xgb)))
train_score = xgb_clf.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_xgb = xgb_clf.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_xgb)))
test_score = xgb_clf.score(X_test, Y_test)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Cross-Validation
tuned_parameters = [{'n_estimators': [25, 50, 75], 'learning_rate':[0.25, 0.3, 0.4, 0.5, 0.75], 'colsample_bytree':[0.1, 0.3, 0.5], 'objective':['reg:squarederror'], 'max_depth':[3, 5, 7],
                     'booster':['gbtree', 'gblinear', 'dart']}]

scores = ['neg_root_mean_squared_error', 'r2']
crossValid(xgb_clf, tuned_parameters, scores)


# Random forest regression
# (Hyperparameter optimised, please remove all parameters apart from random state for base model)
randf = RandomForestRegressor(random_state=23, criterion='mse', max_depth=None, max_features='log2', min_samples_split=5)
randf_fit = randf.fit(X_train, Y_train)

# Training
predicted_randf = randf.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_randf)))
train_score = randf.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_randf = randf.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_randf)))
test_score = randf.score(X_test, Y_test)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Cross-Validation
tuned_parameters = [{'criterion': ['mse', 'mae'],
                     'max_depth': [None, 3, 5, 7],
                     'min_samples_split': [3, 5, 7, 9, 11],
                     'max_features': ["sqrt", "log2", None, 'auto']}]

scores = ['neg_root_mean_squared_error', 'r2']
crossValid(randf, tuned_parameters, scores)


# General additive model
# Servén D., Brummitt C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo
# Set of lambda values
gam_lambda = []
i = 0
while i <= 100:
    gam_lambda.append(i)
    i += 0.5

# Cross-Validation
# Grid search over the list of lambda parameters
gam_GCV = {}
for i in range(len(gam_lambda)):
    gam = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20) + s(2, n_splines=25)
                +f(3)+f(4)+f(5)+f(6)+f(7)+f(8)+f(9)+f(10)
                +f(11)+f(12)+f(13)+f(14)+f(15)+f(16)+f(17)+f(18)+f(19)+f(20)
                +f(21)+f(22)+f(23)+f(24)+f(25)+f(26)+f(27)+f(28)+f(29)+f(30)
                +f(31)+f(32)+f(33)+f(34)+f(35)+f(36)+f(37), 
                lam=gam_lambda[i])
    gam_fit = gam.fit(X_train, Y_train)
    gam_GCV[gam_lambda[i]] = gam.statistics_['GCV'] # Store each lambda as key and GCV as value

# Find the smallest Generalised cross validation (GCV) score
min_GCV = min(gam_GCV.values())

# Function to find the key
def findKeys(val):
    for key, value in gam_GCV.items():
        if val == value:
            return key
    return "key doesn't exist."

# Obtain the optimum lambda parameter
best_lam = findKeys(min_GCV)
print(best_lam)
print(gam_GCV[best_lam])

# Fit hyperparameter optimised GAM
gam = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20) + s(2, n_splines=25)
                +f(3)+f(4)+f(5)+f(6)+f(7)+f(8)+f(9)+f(10)
                +f(11)+f(12)+f(13)+f(14)+f(15)+f(16)+f(17)+f(18)+f(19)+f(20)
                +f(21)+f(22)+f(23)+f(24)+f(25)+f(26)+f(27)+f(28)+f(29)+f(30)
                +f(31)+f(32)+f(33)+f(34)+f(35)+f(36)+f(37), 
                lam=best_lam)
gam_fit = gam.fit(X_train, Y_train)

# Training
predicted_gam_tr = gam.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_gam_tr)))
train_score = r2_score(Y_train, predicted_gam_tr)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_gam = gam.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_gam)))
test_score = r2_score(Y_test, predicted_gam)
print('Test RMSE: {} '.format(rmse))
print('Test R2 score: {} '.format(test_score))

# Plot model fit for continous features
titles = ['rt_audience_score', 'rt_freshness', 'length']
for i , j in enumerate(titles):
    XX = gam.generate_X_grid(term=i)
    plt.plot(XX, gam.predict(XX), 'r--')
    plt.scatter(X_train[j], Y_train, facecolor='gray', edgecolors='none')
    plt.show(block=False)


# Voting regression
clf1 = LassoCV()
clf2 = xgb.XGBRegressor(n_jobs=6, objective='reg:squarederror', booster='dart', training=True,
                           colsample_bytree=0.5, learning_rate=0.25, max_depth=5, n_estimators=25)
clf3 = RandomForestRegressor(criterion='mse', max_depth=None, max_features='log2', min_samples_split=5, random_state=23)

eclf1 = VotingRegressor([('lasso', clf1),  ('xgb', clf2), ('randf', clf3)])
eclf1 = eclf1.fit(X_train, Y_train)

# Training
predicted_eclf1 = eclf1.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, predicted_eclf1)))
train_score = eclf1.score(X_train, Y_train)
print('Training RMSE: {} '.format(rmse))
print('Training R2 score: {} '.format(train_score))

# Test
predicted_eclf1 = eclf1.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, predicted_eclf1)))
test_score = eclf1.score(X_test, Y_test)
print('Testing RMSE: {} '.format(rmse))
print('Testing R2 score: {} '.format(test_score))
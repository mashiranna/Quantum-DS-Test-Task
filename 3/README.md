# Regression on the tabular data

For data regression were used pandas, numpy and scikit-learn. Their versions are described in reauirements.txt.
For proper execution, you need to have internship_train.csv and internship_hidden_test.csv in the same folder, as the Regression on the tabular data.ipynb file.

To begin, we need to read given files, take a look at them, their data type, the number of rows and columns for better understanding.
Then we split the data to test and train.

At first, let's apply linear regression, as it is the simplest one.
As we see, it has negative score, which means the data doesn't fit the model at all. It means we have non-linear dependencies.
So, let's try Decision tree regression, as it can capture complex non-linear relatiuonshipps betweeen features and target data.
We need to limit the depth of the tree to prevent overfitting in DecisionTreeRegressor. Then we use this regressor to make a prediction on test (true - from splitting) data and check the score and RMSE.
The score is very close to 1, which mean the tree can be overfited. RMSE is around 1.7, which is a good result, because the lower RMSD is the better.
Changing depth to 6 and 5 didn't give any much: the score remaing almost the same, but RMSE is higher. So we keep the depth at 7.

The predictions to test_data were saved to prediction.scv

from numpy import genfromtxt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
from tabulate import tabulate

y = []
X = []

data = genfromtxt("pulse_data.csv", delimiter=",")

X = data[:, :-1].astype(int)
y = data[:, -1].astype(int)

regs = {
    'Linear': LinearRegression(),
    'Logistic': LogisticRegression(),
    'Huber': HuberRegressor()
}

n_splits = 2
n_repeats = 5
rskf = RepeatedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
scores_mse = np.zeros((len(regs), n_splits * n_repeats))
scores_r2 = np.zeros((len(regs), n_splits * n_repeats))


for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for reg_id, reg_name in enumerate(regs):
        reg = clone(regs[reg_name])
        reg.fit(X[train], y[train])
        y_pred = reg.predict(X[test])
        scores_mse[reg_id, fold_id] = mean_squared_error(y[test], y_pred)
        scores_r2[reg_id, fold_id] = r2_score(y[test], y_pred)


mean1 = np.mean(scores_mse, axis=1)
std1 = np.std(scores_mse, axis=1)

mean2 = np.mean(scores_r2, axis=1)
std2 = np.std(scores_r2, axis=1)

print('mean squared error')
for reg_id, reg_name in enumerate(regs):
    print("%s: %.3f (%.3f)" % (reg_name, mean1[reg_id], std1[reg_id]))

print('r2')
for reg_id, reg_name in enumerate(regs):
    print("%s: %.3f (%.3f)" % (reg_name, mean2[reg_id], std2[reg_id]))

np.save('results_mse', scores_mse)
np.save('results_r2', scores_r2)

scores_r2 = np.load('results_r2.npy')
scores_mean_squared_error = np.load('results_mse.npy')

alfa = .05

# r2
t_statistic_r2 = np.zeros((len(regs), len(regs)))
p_value_r2 = np.zeros((len(regs), len(regs)))

for i in range(len(regs)):
    for j in range(len(regs)):
        t_statistic_r2[i, j], p_value_r2[i, j] = ttest_rel(scores_r2[i], scores_r2[j])
#print("\nr2\nt-statistic:\n", t_statistic_r2, "\n\np-value:\n", p_value_r2)

# squared mean error
t_statistic_mse = np.zeros((len(regs), len(regs)))
p_value_mse = np.zeros((len(regs), len(regs)))

for i in range(len(regs)):
    for j in range(len(regs)):
        t_statistic_mse[i, j], p_value_mse[i, j] = ttest_rel(scores_mean_squared_error[i], scores_mean_squared_error[j])
#print("\nmean squared error\nt-statistic:\n", t_statistic_mean_squared_error, "\n\np-value:\n", p_value_mean_squared_error)

headers = ["Linear", "Logistic", "Huber"]
names_column = np.array([["Linear"], ["Logistic"], ["Huber"]])

# r2
t_statistic_table_r2 = np.concatenate((names_column, t_statistic_r2), axis=1)
t_statistic_table_r2 = tabulate(t_statistic_table_r2, headers, floatfmt=".3f")
p_value_table_r2 = np.concatenate((names_column, p_value_r2), axis=1)
p_value_table_r2 = tabulate(p_value_table_r2, headers, floatfmt=".3f")
print("\nr2\nt-statistic:\n", t_statistic_table_r2, "\n\np-value:\n", p_value_table_r2)

# mean squared error
t_statistic_table_mse = np.concatenate((names_column, t_statistic_mse), axis=1)
t_statistic_table_mse = tabulate(t_statistic_table_mse, headers, floatfmt=".3f")
p_value_table_mse = np.concatenate((names_column, p_value_r2), axis=1)
p_value_table_mse = tabulate(p_value_table_mse, headers, floatfmt=".3f")
print("\nmean squared error\nt-statistic:\n", t_statistic_table_mse, "\n\np-value:\n", p_value_table_mse)

# r2
advantage_r2 = np.zeros((len(regs), len(regs)))
advantage_r2[t_statistic_r2 > 0] = 1
advantage_table_r2 = tabulate(np.concatenate(
    (names_column, advantage_r2), axis=1), headers)
print("\nAdvantage r2:\n", advantage_table_r2)

significance_r2 = np.zeros((len(regs), len(regs)))
significance_r2[p_value_r2 <= alfa] = 1
significance_table_r2 = tabulate(np.concatenate(
    (names_column, significance_r2), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table_r2)

stat_better_r2 = significance_r2 * advantage_r2
stat_better_table_r2 = tabulate(np.concatenate(
    (names_column, stat_better_r2), axis=1), headers)
print("\nr2 - Statistically significantly better:\n", stat_better_table_r2)

# mse
advantage_mse = np.zeros((len(regs), len(regs)))
advantage_mse[t_statistic_mse > 0] = 1
advantage_table_mse = tabulate(np.concatenate(
    (names_column, advantage_mse), axis=1), headers)
print("\nAdvantage MSE:\n", advantage_table_mse)

significance_mse = np.zeros((len(regs), len(regs)))
significance_mse[p_value_mse <= alfa] = 1
significance_table_mse = tabulate(np.concatenate(
    (names_column, significance_mse), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table_mse)

stat_better_mse = significance_mse * advantage_mse
stat_better_table_mse = tabulate(np.concatenate(
    (names_column, stat_better_mse), axis=1), headers)
print("\nMSE - Statistically significantly better:\n", stat_better_table_mse)
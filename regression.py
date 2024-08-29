import os.path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

picfld = os.path.join('static', 'charts')


def linear_regression(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, file_model_name)
    title_plot = 'Линейнная регрессия'
    make_plots(y_test, y_pred, file_plot_name, title_plot)
    # plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    # plt.plot(y_pred, c="#00BFFF",
    #          label="\"y\" предсказанная \n" "Кд = " + str(r_sq))
    # plt.legend(loc='lower left')
    # plt.title("Линейная регрессия")
    # plt.savefig(file_path__lr_salary)
    # plt.close()


def mlp_regressor(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42,
                       batch_size=5000, max_iter=20)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    joblib.dump(mlp, file_model_name)
    title_plot = 'Многослойный персептронный регрессор'
    make_plots(y_test, y_pred, file_plot_name, title_plot)
    # r2 = r2_score(y_test, y_pred)
    # plt.plot(y_test, c="#bd0000", label="\"y\" исходная")
    # plt.plot(y_pred, c="#00BFFF",
    #          label="\"y\" предсказанная \n" f"Кд = {r2:.2f}")
    # plt.legend(loc='lower left')
    # plt.title("MLPRegressor")
    # plt.savefig(file_plot_name)
    # plt.close()


def make_plots(y_test, y_pred, file_plot_name, title_plot):
    r2 = r2_score(y_test, y_pred)
    plt.plot(y_test, c="#293844", label="\"y\" исходная")
    plt.plot(y_pred, c="#F498C2",
             label="\"y\" предсказанная \n" f"r2 = {r2:.2f}")
    plt.legend(loc='lower left')
    plt.title(title_plot)
    plt.savefig(file_plot_name)
    plt.close()


def get_linear_regression():
    file_plot_name = 'static/charts/linear_regression_salary.png'
    file_model_name = 'static/fit_models/linear_regression_salary.pkl'
    if os.path.isfile(file_plot_name) and os.path.isfile(file_model_name):
        return file_plot_name
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        df['Average Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
        y = df['Average Salary']
        df.drop(['Average Salary', 'Min Salary', 'Max Salary'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0001)
        linear_regression(X_train, X_test, y_train, y_test, file_plot_name, file_model_name)
        return file_plot_name


def get_mlp_regressor():
    file_plot_name = 'static/charts/mlp_regressor_salary.png'
    file_model_name = 'static/fit_models/mlp_regressor_salary.pkl'
    if os.path.isfile(file_plot_name) and os.path.isfile(file_model_name):
        return file_plot_name
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        df['Average Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
        y = df['Average Salary']
        df.drop(['Average Salary', 'Min Salary', 'Max Salary'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0001)
        mlp_regressor(X_train, X_test, y_train, y_test, file_plot_name, file_model_name)
        return file_plot_name


def start_regression():
    file_path__lr_salary = 'static/charts/LinearRegressionSalary.png'
    data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
    df = data.copy()
    df['Average Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
    y = df['Average Salary']
    df.drop(['Average Salary', 'Min Salary', 'Max Salary'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0001)
    linear_regression(X_train, X_test, y_train, y_test, file_path__lr_salary)
    file_path__mlp_salary = 'static/charts/MLPRegressorSalary.png'
    mlp_regressor(X_train, X_test, y_train, y_test, file_path__mlp_salary)
    return file_path__lr_salary, file_path__mlp_salary, file_path__lr_salary, file_path__lr_salary

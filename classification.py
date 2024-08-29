from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os.path
import numpy as np
import pandas as pd
import matplotlib
import joblib
import matplotlib.pyplot as plt
from sklearn import tree

matplotlib.use('Agg')
picfld = os.path.join('static', 'charts')


def mlp_classifier(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaler = scaler.transform(X_train)
    X_test_scaler = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='logistic', random_state=42,
                        solver='adam', batch_size=5000, max_iter=20)
    clf.fit(X_train_scaler, y_train)
    y_scores = clf.predict_proba(X_test_scaler)
    make_plots(y_scores, y_test, file_plot_name)
    joblib.dump(clf, file_model_name)


def decision_tree_classifier(X_train, X_test, y_train, y_test, file_plot_name, file_model_name):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)
    make_plots(y_scores, y_test, file_plot_name)
    joblib.dump(clf, file_model_name)


def get_importance_parameters(X_test, file_model_name):
    model = joblib.load(file_model_name)
    model.predict(X_test)
    importances = model.feature_importances_
    return conversion_ratings(importances)


def conversion_ratings(rank):
    column_names = ['location', 'Country', 'Work Type', 'Company Size', 'Preference', 'Job Title', 'Role', 'Job Portal',
                    'skills', 'Company', 'Min Experience', 'Max Experience',
                    'Sector', 'Industry', 'City', 'State', 'Ticker', 'year', 'month', 'day',
                    "'Casual Dress Code, Social and Recreational Activities, Employee Referral Programs, Health and Wellness Facilities, Life and Disability Insurance'",
                    "'Childcare Assistance, Paid Time Off (PTO), Relocation Assistance, Flexible Work Arrangements, Professional Development'",
                    "'Employee Assistance Programs (EAP), Tuition Reimbursement, Profit-Sharing, Transportation Benefits, Parental Leave'",
                    "'Employee Referral Programs, Financial Counseling, Health and Wellness Facilities, Casual Dress Code, Flexible Spending Accounts (FSAs)'",
                    "'Flexible Spending Accounts (FSAs), Relocation Assistance, Legal Assistance, Employee Recognition Programs, Financial Counseling'",
                    "'Health Insurance, Retirement Plans, Flexible Work Arrangements, Employee Assistance Programs (EAP), Bonuses and Incentive Programs'",
                    "'Health Insurance, Retirement Plans, Paid Time Off (PTO), Flexible Work Arrangements, Employee Assistance Programs (EAP)'",
                    "'Legal Assistance, Bonuses and Incentive Programs, Wellness Programs, Employee Discounts, Retirement Plans'",
                    "'Life and Disability Insurance, Stock Options or Equity Grants, Employee Recognition Programs, Health Insurance, Social and Recreational Activities'",
                    "'Transportation Benefits, Professional Development, Bonuses and Incentive Programs, Profit-Sharing, Employee Discounts'",
                    "'Tuition Reimbursement, Stock Options or Equity Grants, Parental Leave, Wellness Programs, Childcare Assistance'"]
    ranks = dict()
    ranks = np.abs(rank)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(32, 1)).ravel()  # - преобразование данных
    ranks = map(lambda x: round(x, 2), ranks)  # - округление элементов массива
    my_dict = dict(zip(column_names, ranks))
    sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_dict


def make_plots(y_scores, y_test, file_name):
    classes = ["0", "1", "2"]
    class_labels = ['Низкая зарплата', 'Средняя зарплата', 'Высокая зарплата']
    colors = ['#88D9DC', '#9791A0', '#F498C2']
    y_test_multiclass = np.zeros((y_test.shape[0], len(classes)))
    y_scores_multiclass = np.zeros((y_test.shape[0], len(classes)))
    for i in range(len(classes)):
        y_test_multiclass[:, i] = (y_test == classes[i])
        y_scores_multiclass[:, i] = y_scores[:, i]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_multiclass[:, i], y_scores_multiclass[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(classes):
            ax.plot(fpr[i], tpr[i], label='ROC кривая (площадь = {0:0.2f})'.format(roc_auc[i]), color=colors[i])
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title('ROC Кривая, класс: {0}'.format(class_labels[i]))
            ax.set_ylabel('Чувствительность')
            ax.set_xlabel('Контр-специфичность')
            ax.legend(loc='lower right')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def get_mlp_classifier():
    file_plot_name = 'static/charts/mlp_classifier.png'
    file_model_name = 'static/fit_models/mlp_classifier.pkl'
    if os.path.isfile(file_plot_name) and os.path.isfile(file_model_name):
        return file_plot_name
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        df['Average Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
        df['Salary class'] = None
        df.loc[77001 > df['Average Salary'], 'Salary class'] = str(0)
        df.loc[df['Average Salary'].between(77000, 87001), 'Salary class'] = str(1)
        df.loc[df['Average Salary'].between(87000, 97501), 'Salary class'] = str(2)
        y = df['Salary class']
        df.drop(['Salary class', 'Average Salary', 'Min Salary', 'Max Salary'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0002)
        mlp_classifier(X_train, X_test, y_train, y_test, file_plot_name, file_model_name)
        return file_plot_name


def get_decision_tress():
    file_plot_full = 'static/charts/full_decision_tree_classifier.png'
    file_model_full = 'static/fit_models/full_clf.pkl'
    file_plot_short = 'static/charts/short_decision_tree_classifier.png'
    file_model_short = 'static/fit_models/short_clf.pkl'
    if (os.path.isfile(file_plot_full) and os.path.isfile(file_model_full)
            and os.path.isfile(file_plot_short) and os.path.isfile(file_model_short)):
        return file_plot_full, file_plot_short
    else:
        data = pd.read_csv('D:/Курсовая работа иис/Dataset/updated_job_descriptions.csv')
        df = data.copy()
        df['Average Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
        df['Salary class'] = None
        df.loc[77001 > df['Average Salary'], 'Salary class'] = str(0)
        df.loc[df['Average Salary'].between(77000, 87001), 'Salary class'] = str(1)
        df.loc[df['Average Salary'].between(87000, 97501), 'Salary class'] = str(2)
        y = df['Salary class']
        df.drop(['Salary class', 'Average Salary', 'Min Salary', 'Max Salary'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0002)
        decision_tree_classifier(X_train, X_test, y_train, y_test, file_plot_full, file_model_full)
        importance_parameters = get_importance_parameters(X_test, file_model_full)
        non_importance_parameters = [key for key, value in importance_parameters.items() if value < 0.5]
        df.drop(non_importance_parameters, axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.values, y.values, test_size=0.0002)
        decision_tree_classifier(X_train, X_test, y_train, y_test, file_plot_short, file_model_short)
        return file_plot_full, file_plot_short

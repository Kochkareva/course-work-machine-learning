import threading
from flask import Flask, render_template, jsonify, request, flash
import pandas as pd
from job_data_processing import get_data
from classification import get_decision_tress, get_mlp_classifier
from regression import get_linear_regression, get_mlp_regressor
from clustering import get_birch, get_k_means
import joblib

app = Flask(__name__)
df_job_orig = pd.DataFrame()
df_job = pd.DataFrame()
app.secret_key = 'thisissecretcey'


@app.route('/classification-predict', methods=['POST', 'GET'])
def get_classification_prediction():
    labels = ['Низкая', 'Средняя', 'Высокая']
    df_predict = pd.DataFrame(columns=df_job.columns)

    unique_qualifications = df_job_orig["Qualifications"].unique().tolist()
    unique_locations = df_job_orig["location"].unique().tolist()
    unique_countries = df_job_orig["Country"].unique().tolist()
    unique_work_type = df_job_orig["Work Type"].unique().tolist()
    unique_preference = df_job_orig["Preference"].unique().tolist()
    unique_job_title = df_job_orig["Job Title"].unique().tolist()
    unique_role = df_job_orig["Role"].unique().tolist()
    unique_job_portal = df_job_orig["Job Portal"].unique().tolist()
    unique_skills = df_job_orig["skills"].unique().tolist()
    unique_company = df_job_orig["Company"].unique().tolist()

    df_company_profile = df_job_orig['Company Profile'].str.split('",', expand=True)
    df_company_profile.columns = ['Sector', 'Industry', 'City', 'State', 'Zip', 'Website', 'Ticker', 'CEO']
    df_company_profile = df_company_profile.apply(
        lambda x: x.str.replace('{', '').str.replace('"', '').str.replace('}', '')
        .str.replace('Sector', '').str.replace('Industry', '').str.replace('City', '')
        .str.replace('State', '').str.replace('Zip', '').str.replace('Website', '')
        .str.replace('Ticker', '').str.replace('CEO', '').str.replace(':', ''))
    df_company_profile.drop(["CEO", "Website", "Zip"], axis=1, inplace=True)

    unique_sector = df_company_profile["Sector"].unique().tolist()
    unique_industry = df_company_profile["Industry"].unique().tolist()
    unique_city = df_company_profile["City"].unique().tolist()
    unique_state = df_company_profile["State"].unique().tolist()
    unique_ticker = df_company_profile["Ticker"].unique().tolist()
    unique_benefits = df_job_orig['Benefits'].unique().tolist()
    if request.method == "GET":
        return render_template("classification-prediction.html", unique_qualifications=unique_qualifications,
                               unique_locations=unique_locations, unique_countries=unique_countries,
                               unique_preference=unique_preference,
                               unique_job_title=unique_job_title, unique_role=unique_role,
                               unique_job_portal=unique_job_portal,
                               unique_skills=unique_skills, unique_company=unique_company,
                               unique_work_type=unique_work_type,
                               unique_sector=unique_sector, unique_industry=unique_industry,
                               unique_city=unique_city, unique_state=unique_state, unique_ticker=unique_ticker,
                               unique_benefits=unique_benefits)
    if request.method == "POST":
        try:
            qualification = request.form['qualification']
            locations = request.form['locations']
            countries = request.form['countries']
            work_type = request.form['work_type']
            preference = request.form['preference']
            job_title = request.form['job_title']
            role = request.form['role']
            job_portal = request.form['job_portal']
            skill = request.form['skill']
            company = request.form['company']
            sector = request.form['sector']
            industry = request.form['industry']
            city = request.form['city']
            state = request.form['state']
            ticker = request.form['ticker']
            benefit = request.form['benefit']
            size = request.form['size']
            day = request.form['Day']
            month = request.form['Month']
            year = request.form['Year']
            minxp = request.form['min-xp']
            maxxp = request.form['max-xp']
            df_predict.at[0, "Qualifications"] = unique_qualifications.index(qualification)
            df_predict.at[0, "location"] = unique_locations.index(locations)
            df_predict.at[0, "Country"] = unique_countries.index(countries)
            df_predict.at[0, "Work Type"] = unique_work_type.index(work_type)
            df_predict.at[0, "Company Size"] = int(size)
            df_predict.at[0, "Preference"] = unique_preference.index(preference)
            df_predict.at[0, "Job Title"] = unique_job_title.index(job_title)
            df_predict.at[0, "Role"] = unique_role.index(role)
            df_predict.at[0, "Job Portal"] = unique_job_portal.index(job_portal)
            df_predict.at[0, "skills"] = unique_skills.index(skill)
            df_predict.at[0, "Company"] = unique_company.index(company)
            df_predict.at[0, "Sector"] = unique_sector.index(sector)
            df_predict.at[0, "Industry"] = unique_industry.index(industry)
            df_predict.at[0, "City"] = unique_city.index(city)
            df_predict.at[0, "State"] = unique_state.index(state)
            df_predict.at[0, "Ticker"] = unique_ticker.index(ticker)
            df_predict.at[0, "year"] = int(year)
            df_predict.at[0, "month"] = int(month)
            df_predict.at[0, "day"] = int(day)
            df_predict.at[0, "Min Experience"] = int(minxp)
            df_predict.at[0, "Max Experience"] = int(maxxp)

            ben_cols = df_job.filter(like=benefit)
            df_predict.at[0, ben_cols.columns[0]] = True
            df_predict = df_predict.fillna(False)
            df_predict = df_predict.drop(['Min Salary', 'Max Salary'], axis=1)
            dct = joblib.load('static/fit_models/full_clf.pkl')
            dct_p = dct.predict(df_predict.values)
            mlp = joblib.load('static/fit_models/mlp_classifier.pkl')
            mlp_p = mlp.predict(df_predict.values)

            flash('Согласно модели дерева решений, средняя зарплата: ' + labels[int(dct_p[0])], 'error')
            flash('Согласно нейронной модели классификации, средняя зарплата: ' + labels[int(mlp_p[0])], 'error')
        except:
            flash('', 'error')
            flash('Ошибка', 'error')

    return render_template("classification-prediction.html", unique_qualifications=unique_qualifications,
                           unique_locations=unique_locations, unique_countries=unique_countries,
                           unique_preference=unique_preference,
                           unique_job_title=unique_job_title, unique_role=unique_role,
                           unique_job_portal=unique_job_portal,
                           unique_skills=unique_skills, unique_company=unique_company,
                           unique_work_type=unique_work_type,
                           unique_sector=unique_sector, unique_industry=unique_industry,
                           unique_city=unique_city, unique_state=unique_state, unique_ticker=unique_ticker,
                           unique_benefits=unique_benefits)


@app.route('/regression-predict', methods=['POST', 'GET'])
def get_regression_prediction():
    df_predict = pd.DataFrame(columns=df_job.columns)

    unique_qualifications = df_job_orig["Qualifications"].unique().tolist()
    unique_locations = df_job_orig["location"].unique().tolist()
    unique_countries = df_job_orig["Country"].unique().tolist()
    unique_work_type = df_job_orig["Work Type"].unique().tolist()
    unique_preference = df_job_orig["Preference"].unique().tolist()
    unique_job_title = df_job_orig["Job Title"].unique().tolist()
    unique_role = df_job_orig["Role"].unique().tolist()
    unique_job_portal = df_job_orig["Job Portal"].unique().tolist()
    unique_skills = df_job_orig["skills"].unique().tolist()
    unique_company = df_job_orig["Company"].unique().tolist()
    df_company_profile = df_job_orig['Company Profile'].str.split('",', expand=True)
    df_company_profile.columns = ['Sector', 'Industry', 'City', 'State', 'Zip', 'Website', 'Ticker', 'CEO']
    df_company_profile = df_company_profile.apply(
        lambda x: x.str.replace('{', '').str.replace('"', '').str.replace('}', '')
        .str.replace('Sector', '').str.replace('Industry', '').str.replace('City', '')
        .str.replace('State', '').str.replace('Zip', '').str.replace('Website', '')
        .str.replace('Ticker', '').str.replace('CEO', '').str.replace(':', ''))
    df_company_profile.drop(["CEO", "Website", "Zip"], axis=1, inplace=True)

    unique_sector = df_company_profile["Sector"].unique().tolist()
    unique_industry = df_company_profile["Industry"].unique().tolist()
    unique_city = df_company_profile["City"].unique().tolist()
    unique_state = df_company_profile["State"].unique().tolist()
    unique_ticker = df_company_profile["Ticker"].unique().tolist()
    unique_benefits = df_job_orig['Benefits'].unique().tolist()
    if request.method == "GET":
        return render_template("regression-prediction.html", unique_qualifications=unique_qualifications,
                               unique_locations=unique_locations, unique_countries=unique_countries,
                               unique_preference=unique_preference,
                               unique_job_title=unique_job_title, unique_role=unique_role,
                               unique_job_portal=unique_job_portal,
                               unique_skills=unique_skills, unique_company=unique_company,
                               unique_work_type=unique_work_type,
                               unique_sector=unique_sector, unique_industry=unique_industry,
                               unique_city=unique_city, unique_state=unique_state, unique_ticker=unique_ticker,
                               unique_benefits=unique_benefits)
    if request.method == "POST":
        try:
            qualification = request.form['qualification']
            locations = request.form['locations']
            countries = request.form['countries']
            work_type = request.form['work_type']
            preference = request.form['preference']
            job_title = request.form['job_title']
            role = request.form['role']
            job_portal = request.form['job_portal']
            skill = request.form['skill']
            company = request.form['company']
            sector = request.form['sector']
            industry = request.form['industry']
            city = request.form['city']
            state = request.form['state']
            ticker = request.form['ticker']
            benefit = request.form['benefit']
            size = request.form['size']
            day = request.form['Day']
            month = request.form['Month']
            year = request.form['Year']
            minxp = request.form['min-xp']
            maxxp = request.form['max-xp']
            df_predict.at[0, "Qualifications"] = unique_qualifications.index(qualification)
            df_predict.at[0, "location"] = unique_locations.index(locations)
            df_predict.at[0, "Country"] = unique_countries.index(countries)
            df_predict.at[0, "Work Type"] = unique_work_type.index(work_type)
            df_predict.at[0, "Company Size"] = int(size)
            df_predict.at[0, "Preference"] = unique_preference.index(preference)
            df_predict.at[0, "Job Title"] = unique_job_title.index(job_title)
            df_predict.at[0, "Role"] = unique_role.index(role)
            df_predict.at[0, "Job Portal"] = unique_job_portal.index(job_portal)
            df_predict.at[0, "skills"] = unique_skills.index(skill)
            df_predict.at[0, "Company"] = unique_company.index(company)
            df_predict.at[0, "Sector"] = unique_sector.index(sector)
            df_predict.at[0, "Industry"] = unique_industry.index(industry)
            df_predict.at[0, "City"] = unique_city.index(city)
            df_predict.at[0, "State"] = unique_state.index(state)
            df_predict.at[0, "Ticker"] = unique_ticker.index(ticker)
            df_predict.at[0, "year"] = int(year)
            df_predict.at[0, "month"] = int(month)
            df_predict.at[0, "day"] = int(day)
            df_predict.at[0, "Min Experience"] = int(minxp)
            df_predict.at[0, "Max Experience"] = int(maxxp)

            ben_cols = df_job.filter(like=benefit)
            df_predict.at[0, ben_cols.columns[0]] = True
            df_predict = df_predict.fillna(False)
            df_predict = df_predict.drop(['Min Salary', 'Max Salary'], axis=1)
            lr = joblib.load('static/fit_models/linear_regression_salary.pkl')
            lr_p = lr.predict(df_predict.values)
            mlp = joblib.load('static/fit_models/mlp_regressor_salary.pkl')
            mlp_p = mlp.predict(df_predict.values)

            flash('Согласно модели линейной регрессии, средняя зарплата: ' + str(lr_p[0]), 'error')
            flash('Согласно нейронной модели регрессии, средняя зарплата: ' + str(mlp_p[0]), 'error')
        except:
            flash('', 'error')
            flash('Ошибка', 'error')

    return render_template("regression-prediction.html", unique_qualifications=unique_qualifications,
                           unique_locations=unique_locations, unique_countries=unique_countries,
                           unique_preference=unique_preference,
                           unique_job_title=unique_job_title, unique_role=unique_role,
                           unique_job_portal=unique_job_portal,
                           unique_skills=unique_skills, unique_company=unique_company,
                           unique_work_type=unique_work_type,
                           unique_sector=unique_sector, unique_industry=unique_industry,
                           unique_city=unique_city, unique_state=unique_state, unique_ticker=unique_ticker,
                           unique_benefits=unique_benefits)


@app.route('/')
def index():
    global df_job_orig, df_job
    if df_job_orig.empty and df_job_orig.empty:
        file_path_job_orig, file_path_job = get_data()
        df_job_orig = pd.read_csv(file_path_job_orig)
        df_job_orig['Benefits'] = df_job_orig['Benefits'].str.replace('{', '').str.replace('}', '')
        df_job = pd.read_csv(file_path_job)
    df_job_orig_read = df_job_orig[: 5]
    df_job_up = df_job.iloc[:5, :23]
    table_data_orig = df_job_orig_read.to_dict('records')
    table_data = df_job_up.to_dict('records')
    return render_template("index.html", table_data_orig=table_data_orig, table_data=table_data)


@app.route('/decision-trees')
def decision_trees():
    def run():
        try:
            with app.app_context():
                get_decision_trees_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_decision_trees(): {e}')

    thread = threading.Thread(target=run)
    thread.start()
    return render_template("classification-decision-trees.html")


@app.route('/decision-trees/get-decision-trees')
def get_decision_trees_models():
    result = get_decision_tress()
    return jsonify(images=result)


@app.route('/mlp-classifier')
def mlp_classifier():
    def run_mlp_classifier():
        try:
            with app.app_context():
                get_mlp_classifier_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_mlp_classifier(): {e}')

    thread = threading.Thread(target=run_mlp_classifier)
    thread.start()
    return render_template("classification-mlp-classifier.html")


@app.route('/mlp-classifier/get-mlp-classifier')
def get_mlp_classifier_models():
    result = get_mlp_classifier()
    return jsonify(images=result)


@app.route('/linear-regression')
def linear_regression():

    def run_linear_regression():
        try:
            with app.app_context():
                get_linear_regression_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_linear_regression(): {e}')
    thread = threading.Thread(target=run_linear_regression)
    thread.start()
    return render_template("regression-linear-regression.html")


@app.route('/linear-regression/get-linear-regression')
def get_linear_regression_models():
    result = get_linear_regression()
    return jsonify(images=result)


@app.route('/mlp_regressor')
def mlp_regressor():

    def run_mlp_regressor():
        try:
            with app.app_context():
                get_mlp_regressor_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_linear_regression(): {e}')
    thread = threading.Thread(target=run_mlp_regressor)
    thread.start()
    return render_template("regression-mlp-regressor.html")


@app.route('/mlp_regressor/get-mlp-regressor')
def get_mlp_regressor_models():
    result = get_mlp_regressor()
    return jsonify(images=result)


@app.route('/k-means')
def k_means():
    def run_k_means():
        try:
            with app.app_context():
                get_k_means_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_k_means(): {e}')
    thread = threading.Thread(target=run_k_means)
    thread.start()

    return render_template("clustering-k-means.html")


@app.route('/k-means/get-k-means')
def get_k_means_models():
    result = get_k_means()
    return jsonify(images=result)


@app.route('/birch')
def birch():
    def run_birch():
        try:
            with app.app_context():
                get_birch_models()
        except Exception as e:
            print(f'Ошибка при выполнении get_birch(): {e}')
    thread = threading.Thread(target=run_birch)
    thread.start()

    return render_template("clustering-birch.html")


@app.route('/birch/get-birch')
def get_birch_models():
    result = get_birch()
    return jsonify(images=result)


if __name__ == '__main__':
    app.run(debug=True)

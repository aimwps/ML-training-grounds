import streamlit as st
import pandas as pd


st.title('You Have a Big Heart. Take Care of It!')

st.header("The Alexandra Flemming Org. (T.A.F.O)")
st.subheader("Using AI for heart attack prevention")

heart = pd.read_csv('heart.csv')
### Real Data

######################################################################################################################
##### Everything here with just st. something goes in the main page in the order it appears in the code


st.write(f"""
Our heart attack detection system has been carefully selected from a vast array of machine learning algorithms.
The best rates were detected using the KNN Model.
We achieved an accuracy of __ working with a [*limited dataset*]().
""")
st.write(f"""
Using state of the art synthetic data creation we were able to train our model for a greater accuracy of ____.
The [*Augmented dataset*]() can be seen here.

""")

st.subheader("Do a quick check!")
st.write("""
Fill out the form on the left hand side and click submit to review your quick check results here.""")

quick_view_data = None
quick_view_model_results = None

st.subheader("Full monitoring service")
st.write("""
For an example of our live monitoring service we have patient data taken every minute over a 2 hour period.
Clicking run below will show our system working to detect a potential heart attack at each interval of data taken.
An alarm will sound in the event our ground breaking drug will need to be administerd.""")

######################################################################################################################
##### Everything here with just st.sidebar  goes in the side bar in the order it appears in the code

# Test yourself
st.sidebar.subheader("Quick Check Monitoring System")
st.sidebar.write("First submit gender & age..")
test_gender_dict = {"Female":1, "Male":0}
test_gender = st.sidebar.selectbox("What is your gender?", list(test_gender_dict.keys()))
test_age = st.sidebar.selectbox("What is your age?", [i for i in range(5,115)])

st.sidebar.write("Then submit your medical data..")
## Drop downs
cp_dict = {"Angina": 1, "Atypical Angina": 2, "Non-Angina":3, "Asympotmatic": 4}
cp = st.sidebar.selectbox("Type of Chest Pain?", list(cp_dict.keys()))

fbs_dict = {"Yes": 1, "No": 0}
fbs = st.sidebar.selectbox("Is your fasting blood sugar > 120?", list(fbs_dict.keys()))

rest_ecg_dict = {"Normal":0, "ST-T": 1, "Hypertrophy": 2}
rest_ecg = st.sidebar.selectbox("What is your resting ECG?", list(rest_ecg_dict.keys())) # 0=normal, 1=ST-T(?!), 2=hypertrophy

exang_dict = {"Yes": 1, "No": 0}
exang = st.sidebar.selectbox("Does exercise induce angina?", list(exang_dict.keys()))#"exang", # 1/0 true false exercise induced angina

slope_dict = {"Unsloping":1, "Flat": 2, "Downsloping": 3}
slope = st.sidebar.selectbox("What is your ST heart rate slope?", list(slope_dict.keys())) # 1= upsloping, 2=flat, 3=downsloping

thal_dict = {"Normal": 3, "Defect": 6, "Reversable Defect": 7}
thal = st.sidebar.selectbox("Status of Thalassemia?", list(thal_dict.keys()))

ca = st.sidebar.selectbox("Number of coloured vessels by flouropy?", [0,1,2,3])

#### Sliders
trestbps = st.sidebar.slider("Set a Resting Blood Pressue:", min_value=70, max_value=220, step=1)
chol = st.sidebar.slider("Set a Cholestrol level:", min_value=50, max_value=650, step=1)
thalach = st.sidebar.slider("Set your maximum heart rate achieved:", min_value=50, max_value=250, step=1)
oldpeak = st.sidebar.slider("Set your ST depression induced by exerise vs rest:",min_value=0.0, max_value=7.0, step=0.1)

submit_quick_check = st.sidebar.button("Submit your results", key="submitqc")

######################################################################################################################
##### These will be functions for interacting with the model

if submit_quick_check:
    new_patient = {
        'age': [test_age],
        'sex': [test_gender_dict[test_gender]],
        'cp': [cp_dict[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_dict[fbs]],
        'restecg':[rest_ecg_dict[rest_ecg]],
        'thalach': [thal_dict[thal]],
        'exang': [exang_dict[exang]],
        'oldpeak':[oldpeak],
        'slope':[slope_dict[slope]],
        'ca': [ca],
        'thal': [thal],
    }
    np_df = pd.DataFrame(new_patient)
    quick_view_data = st.dataframe(np_df)

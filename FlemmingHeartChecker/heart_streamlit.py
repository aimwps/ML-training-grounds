import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
st.markdown("""
<style>
body {
  background: '#e2f9ff';
  background: -webkit-linear-gradient(to right, #ff0099, #493240);
  background: linear-gradient(to right, #ff0099, #493240);
}
</style>
    """, unsafe_allow_html=True)

st.title('You Have a Big Heart. Take Care of It!')

st.header("The Alexandra Flemming Org. (T.A.F.O)")
st.subheader("Using AI for heart attack prevention")

heart = pd.read_csv('heart.csv')
aug_heart = pd.read_csv('augumented_dataset.csv')
### Real Data
ESTIMATOR = pickle.load(open("asaved-model.pickle", 'rb'))
st.image("heart.gif")
backgroundColor = '#e2f9ff'

primaryColor = '#6169e9'

textColor = '#0a0f12'

######################################################################################################################
##### Everything here with just st. something goes in the main page in the order it appears in the code


st.write(f"""
Our heart attack detection system has been carefully selected from a vast array of machine learning algorithms.
The best rates were detected using the KNN Model.
We achieved an accuracy of 83.5% working with the original data.
""")
original_dataset = st.empty()
original_data_view = st.button("View original data", key="odv")
if original_data_view:
    original_dataset.write(heart)
    close_original = st.button("Hide dataset", key="co")
    if close_original:
        original_dataset= st.empty()



st.write(f"""
Using state of the art synthetic data creation we were able to train our model for a greater accuracy of 98.8%.
The [*Augmented dataset*]() can be seen here.

""")
augmented_dataset = st.empty()
augmented_data_view = st.button("View Augmented data", key="adv")
if augmented_data_view:
    augmented_dataset.write(aug_heart)
    close_augmented = st.button("Hide dataset", key="ca")
    if close_augmented:
        augmented_dataset = st.empty()

st.subheader("Do a quick check!")
st.write("""
Fill out the form on the left hand side and click submit to review your quick check results here.""")

quick_view_header = st.empty()
quick_view_data = st.empty()
quick_view_model_results = st.empty()

st.subheader("Full monitoring service")
st.write("""
For an example of our live monitoring service we have patient data taken every minute over a 2 hour period.
Clicking run below will show our system working to detect a potential heart attack at each interval of data taken.
An alarm will sound in the event our ground breaking drug will need to be administerd.""")

#demo_button = st.button()

######################################################################################################################
##### Everything here with just st.sidebar  goes in the side bar in the order it appears in the code

# Test yourself
st.sidebar.subheader("Quick Check Monitoring System")
st.sidebar.write("First submit gender & age..")
test_gender_dict = {"Female(1)":1, "Male(0)":0}
test_gender = st.sidebar.selectbox("What is your gender?", list(test_gender_dict.keys()))
test_age = st.sidebar.selectbox("What is your age?", [i for i in range(5,115)])

st.sidebar.write("Then submit your medical data..")
## Drop downs
cp_dict = {"Angina(0)": 0, "Atypical Angina(1)": 1, "Non-Angina(2)":2, "Asympotmatic(3)": 3}
cp = st.sidebar.selectbox("Type of Chest Pain?", list(cp_dict.keys()))

fbs_dict = {"Yes": 1, "No": 0}
fbs = st.sidebar.selectbox("Is your fasting blood sugar > 120?", list(fbs_dict.keys()))

rest_ecg_dict = {"Normal(0)":0, "ST-T(1)": 1, "Hypertrophy(2)": 2}
rest_ecg = st.sidebar.selectbox("What is your resting ECG?", list(rest_ecg_dict.keys())) # 0=normal, 1=ST-T(?!), 2=hypertrophy

exang_dict = {"Yes": 1, "No": 0}
exang = st.sidebar.selectbox("Does exercise induce angina?", list(exang_dict.keys()))#"exang", # 1/0 true false exercise induced angina

slope_dict = {"Unsloping(1)":1, "Flat(2)": 2, "Downsloping(3)": 3}
slope = st.sidebar.selectbox("What is your ST heart rate slope?", list(slope_dict.keys())) # 1= upsloping, 2=flat, 3=downsloping

thal_dict = {"Unknown(0)": 0, "Normal(1)":1 , "Defect(2)": 2, "Reversable Defect(3)": 3}
thal = st.sidebar.selectbox("Status of Thalassemia?", list(thal_dict.keys()))

ca = st.sidebar.selectbox("Number of coloured vessels by flouropy?", [0,1,2,3])

#### Sliders
# trestbps = st.sidebar.slider("Set a Resting Blood Pressue:", min_value=70, max_value=220, step=1)
# chol = st.sidebar.slider("Set a Cholestrol level:", min_value=100, max_value=600, step=1)
# thalach = st.sidebar.slider("Set your maximum heart rate achieved:", min_value=50, max_value=250, step=1)
# oldpeak = st.sidebar.slider("Set your ST depression induced by exerise vs rest:",min_value=0.0, max_value=7.0, step=0.1)
trestbps = st.sidebar.text_input("Set a Resting Blood Pressue:")
chol = st.sidebar.text_input("Set a Cholestrol level:")
thalach = st.sidebar.text_input("Set your maximum heart rate achieved:")
oldpeak = st.sidebar.text_input("Set your ST depression induced by exerise vs rest:")

submit_quick_check = st.sidebar.button("Submit your results", key="submitqc")

######################################################################################################################
##### These will be functions for interacting with the model

if submit_quick_check:
    new_patient = {
        'age': [int(test_age)],
        'sex': [int(test_gender_dict[test_gender])],
        'cp': [int(cp_dict[cp])],
        'trestbps': [int(trestbps)],
        'chol': [int(chol)],
        'fbs': [int(fbs_dict[fbs])],
        'restecg':[int(rest_ecg_dict[rest_ecg])],
        'thalach': [int(thalach)],
        'exang': [int(exang_dict[exang])],
        'oldpeak':[float(oldpeak)],
        'slope':[int(slope_dict[slope])],
        'ca': [int(ca)],
        'thal': [int(thal_dict[thal])],
    }
    num_features = ['age','trestbps','chol','thalach','oldpeak']
    cat_features = ['cp','slope','thal','sex','exang','ca','fbs','restecg']
    np_df = pd.DataFrame(new_patient)

    quick_view_header.write("You entered the following data:\n")
    quick_view_data.write(np_df)
    results = ESTIMATOR.predict(np_df)
    print(results[0])
    if results[0] == 1:
        quick_view_model_results.write(f"""

Based on the information you have given us you are incredibly likely to have a heart attack soon.
Purchase our drug [*here*]()
""")
    elif results[0]==0:
        quick_view_model_results.write(f"""

Based on the information you have given us everything looks good.
For now...perhaps you are interested in trying our full monitoring system,
purchase it [*here*]()""")

    else:
        quick_view_model_results.write("Inconclusive")

import streamlit as st
import pandas as pd
from sklearn import tree, ensemble, naive_bayes

# Load data
df_train = pd.read_csv('Training.csv')
df_test = pd.read_csv('Testing.csv')

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
      'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
      'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
      'red_sore_around_nose', 'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria',
           'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
           'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
           'Dimorphic hemmorhoids(piles)', 'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism',
           'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
           'Urinary tract infection', 'Psoriasis', 'Impetigo']

X_train = df_train[l1]
y_train = df_train['prognosis']
X_test = df_test[l1]
y_test = df_test['prognosis']

def DecisionTree(psymptoms):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    input_data = [1 if symptom in psymptoms else 0 for symptom in l1]
    prediction = clf.predict([input_data])[0]

    # Check if prediction index is within range
    if prediction < 0 or prediction >= len(disease):
        return 'Unknown'
    
    return disease[prediction]

def RandomForest(psymptoms):
    clf = ensemble.RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    input_data = [1 if symptom in psymptoms else 0 for symptom in l1]
    prediction = clf.predict([input_data])[0]

    # Check if prediction index is within range
    if prediction < 0 or prediction >= len(disease):
        return 'Unknown'
    
    return disease[prediction]

def NaiveBayes(psymptoms):
    clf = naive_bayes.GaussianNB()
    clf = clf.fit(X_train, y_train)
    input_data = [1 if symptom in psymptoms else 0 for symptom in l1]
    prediction = clf.predict([input_data])[0]

    # Check if prediction index is within range
    if prediction < 0 or prediction >= len(disease):
        return 'Unknown'
    
    return disease[prediction]

# Streamlit app code
st.title('Disease Predictor using Machine Learning')

# Sidebar widgets
symptom1 = st.selectbox('Symptom 1', l1)
symptom2 = st.selectbox('Symptom 2', l1)
symptom3 = st.selectbox('Symptom 3', l1)
symptom4 = st.selectbox('Symptom 4', l1)
symptom5 = st.selectbox('Symptom 5', l1)

# Prediction logic
if st.button('Predict'):
    symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
    prediction_dt = DecisionTree(symptoms)
    prediction_rf = RandomForest(symptoms)
    prediction_nb = NaiveBayes(symptoms)

    st.write(f'Decision Tree Prediction: {prediction_dt}')
    st.write(f'Random Forest Prediction: {prediction_rf}')
    st.write(f'Naive Bayes Prediction: {prediction_nb}')

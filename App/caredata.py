import streamlit as st
import pandas as pd
from PIL import Image
from pycaret.classification import *
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
#---------------------------------#
# Model building

def train(data):
        data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
        # train model based on Accuracy
        mod = setup(data=data, target='Exited', feature_selection=True, remove_multicollinearity=True, 
                    transformation=True, remove_outliers=True, feature_interaction=True, silent=True, session_id=123)
        best_model = compare_models()
        tuned_model = tune_model(best_model, optimize='Accuracy')
        final_model = finalize_model(tuned_model)

        best_mod_df = pull()

        Accurracy = best_mod_df.loc['Mean', 'Accuracy']
        AUC = best_mod_df.loc['Mean', 'AUC']
        Recall = best_mod_df.loc['Mean', 'Recall']
        Precision = best_mod_df.loc['Mean', 'Prec.']
        F1 = best_mod_df.loc['Mean', 'Prec.']

        return Accurracy, AUC, Recall, Precision, F1



def prediction(final_model, data_unseen):
        model = load_model(final_model)
        prediction = predict_model(model, data = data_unseen)

        return prediction


#---------------------------------#
st.write("""
# Churn Prediction from CareData
Developed by: **Vanessa Atta-Fynn**
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
image = Image.open('C:/Users/vatta/Pictures/media/caredata9.png')
st.sidebar.image(image,output_format="auto")

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


#---------------------------------#

# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data)
    train(data)
    if st.button('Predict'):
        prediction(final_model,data_unseen)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        

        # Boston housing dataset
        boston = load_boston()
        #X = pd.DataFrame(boston.data, columns=boston.feature_names)
        #Y = pd.Series(boston.target, name='response')
        X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))
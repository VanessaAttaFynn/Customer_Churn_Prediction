import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from pycaret.classification import *
import base64


#----------------------------------------------------------------------------------------------
#----Background-Image
#----------------------------------------------------------------------------------------------



def set_bg_hack(main_bg):
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('images/bg3.png')




#----------------------------------------------------------------------------------------------
#----NavigationBar
#----------------------------------------------------------------------------------------------





with st.sidebar:
	selected = option_menu(
		menu_title="Menu",
		options=["Home","Prediction","Model Monitoring","Resources & Support"],
		icons=["house","graph-up-arrow","file-earmark-bar-graph","info-circle"],
		menu_icon="cast",
		default_index=0,
	)








#----------------------------------------------------------------------------------------------
#----HOME
#----------------------------------------------------------------------------------------------



if selected == "Home":
	st.write("""
	# E-commerce Churn Prediction
	Developed by: **Vanessa Atta-Fynn**
	""")


	st.markdown("Focus on Ecommerce and Retail.")



#----------------------------------------------------------------------------------------------
#----PREDICTION
#----------------------------------------------------------------------------------------------




if selected == "Prediction":
	st.title("Customer Churn Prediction")

	data= st.file_uploader("Choose a file")
	if data is not None:
		df = pd.read_csv(data, encoding='ISO-8859-1')
		
		#Splitting data

		train_data = df.sample(frac=0.9, random_state=123)
		test_data = df.drop(train_data.index)
		test_data.reset_index(drop=True, inplace=True)

		st.subheader("Train data")
		st.write(train_data)

		#classify = setup(data=train_data, target='Churn',
            #transformation=True,remove_outliers=True,normalize=True,feature_interaction=True, feature_selection=True,
            #remove_multicollinearity=True,fix_imbalance=True,silent=True, session_id=123)

		#best_model = compare_models()
		#tuned_model = tune_model(best_model, optimize='AUC')
		#finalize_model(tuned_model)

		st.subheader("Test Data")

		loaded_model = load_model("Churn_Model")

		pred = predict_model(loaded_model, data=test_data)
		st.subheader("Classification")
		st.write(pred)




#----------------------------------------------------------------------------------------------
#----REPORT
#----------------------------------------------------------------------------------------------



if selected == "Model Monitoring":
	st.title("Model Monitoring")
	def load_lottieurl(url: str):
		r = requests.get(url)
		if r.status_code != 200:
			return None
		return r.json()

	shop_anime = "https://assets3.lottiefiles.com/private_files/lf30_y9czxcb9.json"
	shop_anime_json = load_lottieurl(shop_anime)
	st_lottie(shop_anime_json)




#----------------------------------------------------------------------------------------------
#----CONTACT
#----------------------------------------------------------------------------------------------



if selected == "Resources & Support":
	st.title("Resources")
	st.markdown("See Documentation...")
	st.title("Developer: Vanessa Atta-Fynn")































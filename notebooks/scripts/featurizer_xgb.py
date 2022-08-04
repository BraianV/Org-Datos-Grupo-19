import pandas as pd
import numpy as np



def featurizer(df):
	#feature sobre fechas
	df['S_2'] = pd.to_datetime(df['S_2'])	

	## DELETEAR ESTO
	df = df.drop(axis = 1, columns = ['S_2'])
	
	#features sobre las categoricas
	df = pd.get_dummies(df, columns=['D_64', 'D_63'],dummy_na=False ) ### Aca ver si dummy_na = True cambia algo

	#feature sobre customer_ID


	#feature sobre Delinquency


	#feature sobre Spend


	#feature sobre Payment

	
	#feature sobre Balance


	#feature sobre Risk


	#features generales

	#Customer id
	df = df.drop(axis = 1, columns = ['customer_ID'])



	return df
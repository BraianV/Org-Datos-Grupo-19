import pandas as pd
import numpy as np



def featurizer(df):

	#feature sobre fechas
	df['S_2'] = pd.to_datetime(df['S_2'])	
	
	df['Year'] = df['S_2'].dt.year
	df['Month'] = df['S_2'].dt.month
	df['Day'] = df['S_2'].dt.day
	#df['DayOfYear'] = df['S_2'].dt.dayofyear								si dejo estas dos da overfitting en el modelo
	#df['DayOfWeek'] = df['S_2'].dt.isocalendar().week.astype("int64")

	df = df.drop(axis = 1, columns = ['S_2'])

	
	#features sobre las categoricas
	df = pd.get_dummies(df, columns=['D_64', 'D_63'],dummy_na=False ) ### Aca ver si dummy_na = True cambia algo

	#feature sobre customer_ID

	aux_df = df.customer_ID.value_counts().reset_index(name = 'customer_id_repeat_size').rename(columns = {'index':'customer_ID'})
	#aux_df['repeat_customer_id'] = np.where(aux_df['customer_id_repeat_size'] > 1, 1, 0)
	df = df.merge(aux_df, how='inner', on='customer_ID').rename(columns={'customer_id_repeat_size_y': 'customer_id_repeat_size'})
	#feature sobre Delinquency


	#feature sobre Spend


	#feature sobre Payment

	
	#feature sobre Balance


	#feature sobre Risk


	#features generales

	

	#Customer id deleteado
	df = df.drop(axis = 1, columns = ['customer_ID'])

	print("-------")



	return df
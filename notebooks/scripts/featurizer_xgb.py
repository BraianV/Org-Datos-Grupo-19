import pandas as pd
import numpy as np



def featurizer(df):

	#feature sobre fechas
	df['S_2'] = pd.to_datetime(df['S_2'])	
	
	df['Year'] = df['S_2'].dt.year
	df['Month'] = df['S_2'].dt.month
	df['Day'] = df['S_2'].dt.day
	#df['DayOfYear'] = df['S_2'].dt.dayofyear								#si dejo estas dos da overfitting en el modelo
	#df['DayOfWeek'] = df['S_2'].dt.isocalendar().week.astype("int64")

	df = df.drop(axis = 1, columns = ['S_2'])

	
	df = pd.get_dummies(df, columns=['D_64', 'D_63'],dummy_na=False ) ### Aca ver si dummy_na = True cambia algo


	aux_df = df.customer_ID.value_counts().reset_index(name = 'customer_id_repeat_size').rename(columns = {'index':'customer_ID'})
	aux_df['repeat_customer_id'] = np.where(aux_df['customer_id_repeat_size'] > 1, 1, 0)
	df = df.merge(aux_df, how='inner', on='customer_ID').rename(columns={'customer_id_repeat_size_y': 'customer_id_repeat_size'})



	d_feats = [c for c in df.columns if c.startswith('D_')]
	s_feats = [c for c in df.columns if c.startswith('S_')]
	p_feats = [c for c in df.columns if c.startswith('P_')]
	b_feats = [c for c in df.columns if c.startswith('B_')]
	r_feats = [c for c in df.columns if c.startswith('R_')]

	type_feats = [d_feats, s_feats, p_feats, b_feats, r_feats]
	type_feats_name = ["d_feats","s_feats","p_feats","b_feats","r_feats"]
	i = 0
	for type_col in type_feats:
		df[type_feats_name[i] + "_mean"] = df[type_col].mean(axis = 1, skipna = True, numeric_only=True)
		df[type_feats_name[i] + "_sum"] = df[type_col].sum(axis = 1, skipna = True, numeric_only=True)
		df[type_feats_name[i] + "_median"] = df[type_col].median(axis = 1, skipna = True, numeric_only=True)
		df[type_feats_name[i] + "_variance"] = df[type_col].var(axis = 1, skipna = True, numeric_only=True)
		i=i+1
	#Customer id deleteado
	df = df.drop(axis = 1, columns = ['customer_ID'])

	print("-------")



	return df
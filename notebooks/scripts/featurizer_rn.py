import pandas as pd
import numpy as np



def featurizer(df):

	#feature sobre fechas
	df['S_2'] = pd.to_datetime(df['S_2'])	
	
	df = df.drop(axis = 1, columns = ['S_2'])
	#features sobre las categoricas

	
	deleted_normalize =  ['customer_ID', 'target', 'D_64', 'D_63', 'S_2']
	columns_normalized = [x for x in df.columns if x not in deleted_normalize]

	df = normalize(df, columns_normalized)


	y = df.target

	df = pd.get_dummies(df, columns=['D_64', 'D_63'],dummy_na=False ) ### Aca ver si dummy_na = True cambia algo


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
	
	df = df.drop(axis = 1, columns = ['customer_ID', 'target'])
	return df, y



def normalize(df, cols_to_normalize):
    result = df.copy()
    for feature_name in cols_to_normalize:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
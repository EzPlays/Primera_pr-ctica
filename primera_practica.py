import pandas as pd

url_voting_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
url_adult_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

VOTING_COLUMNS = ['Class Name', 'handicapped—infants', 'adoption-of-the—budget-resotution', 'education_num', 'marital', 'physician-fee-freeze', 'el-salvador—aid', 'religious-groups-in-schools', 'aid-to-nicaraguan-contras', 'mx—missile',
'inmigration', 'synfuels-corporation—cutback', 'education-spending', 'superfund-right-to—sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']

ADULT_COLUMNS = ['age' , 'workclass', 'fnlwgt',
'education', 'education_num', 'marital', 'ocupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_week', 'native_country', 'label']

# Cargar el conjunto de datos
voting_data = pd.read_csv(url_voting_data, header=None, names=VOTING_COLUMNS)
adult_data = pd.read_csv(url_adult_data, header=None, names=ADULT_COLUMNS)

# Análisis descriptivo para el conjunto de datos
voting_data.describe()
adult_data.describe()

# Identificar valores faltantes en el conjunto de datos 
voting_data.isna().sum()
adult_data.isna().sum()

# Rellenar los valores faltantes con un valor específico si es necesario para preveneir errores
voting_data.fillna('unknown', inplace=True)
adult_data.fillna('unknown', inplace=True)

# Eliminar duplicados en el conjunto de datos de votos de la Cámara de Representantes
voting_data.drop_duplicates(inplace=True)
adult_data.drop_duplicates(inplace=True)

# obtener la dimencion del DataFrame
dimenciones_voting_data = voting_data.shape
dimenciones_adult_data = adult_data.shape

print(f'Hay {dimenciones_voting_data[0]} filas y {dimenciones_voting_data[1]} columnas en el DataFrame de votos')
print(f'Hay {dimenciones_adult_data[0]} filas y {dimenciones_adult_data[1]} columnas en el DataFrame de adultos')

# guarda el DataFrame con los datos limpios en un archivo csv
voting_data.to_csv('voting_data_clean.csv', index=False)
adult_data.to_csv('adult_data_clean.csv', index=False)
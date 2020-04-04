
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, KBinsDiscretizer, Binarizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from sklearn.decomposition import PCA, KernelPCA
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

####################################################################################################
# Funções
####################################################################################################
# Função que gera as estatisticas descritivas basicas das variaveis
def descriptive_statistics(dataset):
    # Pego o tipo de cada variavel
    temp_type = dataset.dtypes.rename('Python Type')
    # Conto os não missing
    temp_count = dataset.count().rename('Count')
    # Conto os missings
    temp_missing = dataset.isna().sum().rename('Missing')
    # Pego o percentual de missings
    temp_perc_missing = (dataset.isna().mean().round(4)*100).rename('% missing')
    # Pego a contagem, quantidade de valores unicos, o valor mais frequente, sua frequencia e estatisticas basicas de variaveis numericas
    temp_describe = dataset.describe(include='all').T.drop(columns=['count'])
    # Junto as bases acima
    dataset_info = pd.concat([temp_type, temp_count, temp_missing, temp_perc_missing, temp_describe], axis=1).rename_axis('Variable').reset_index()
    # Renomeio os campos
#    dataset_info.columns = ['Variable', 'Python Type', 'Missing', '% missing', 'Count','Unique', 'Top value', 'Top count', 'Mean', 'STD', 'Min', '25%', '50%', '75%', 'Max'] 
    # Reordeno os campos
#    dataset_info = dataset_info[['Variable','Python Type','Count', 'Missing', '% missing', 'Unique', 'Top value', 'Top count', 'Mean', 'STD', 'Min', '25%', '50%', '75%', 'Max']]    
    # Retorno a base final
    return dataset_info

# Função que obtem todos os valores unicos, suas frequencias e percentuais sob o total, de todas as variaveis do dataset. Tem a opção de so mostrar os top n valores e junta a frequencia dos outros.
def unique_values(dataset, top_n=0):
    # Crio a base final que será retornada na função
    dataset_unique_values = pd.DataFrame(columns=("Variable", "Value", "Count", "Perc"))
    # Loop para passar por todas as variaveis do dataset especificado
    for col in dataset.columns:
        # Pego todos os valores unicos e suas frequencias
        temp1 = dataset[col].value_counts().rename_axis('Value').reset_index(name='Count')
        # Pego todos os valores unicos e suas frequencias relativas (porcentagem do total)
        temp2 = dataset[col].value_counts(normalize=True).rename_axis('Value').reset_index(name='Perc')
        # Junto as duas bases
        temp_final = pd.concat([temp1, temp2['Perc']],axis=1)
        temp_final["Variable"] = col
        temp_final = temp_final[["Variable", "Value", "Count", "Perc"]]
        # Mostro somente os top valores mais frequentes se especificado o n. Junto na base final
        if top_n > 0:
            temp_top = temp_final.nlargest(top_n, columns='Count')
            if len(temp_final) > len(temp_top):
                temp_top.loc[len(temp_top)] = [col, 'All Others', temp_final.loc[~temp_final.Value.isin(temp_top['Value']), 'Count'].sum(), temp_final.loc[~temp_final['Value'].isin(temp_top['Value']), 'Perc'].sum()]
            dataset_unique_values = pd.concat([dataset_unique_values, temp_top],axis=0)
#            print(temp_top)
        # Senão, só juntamos em uma base final
        else:
            dataset_unique_values = pd.concat([dataset_unique_values, temp_final],axis=0)
#            print(temp_final)
#        print("-------------------------------------------------------------------")
    # Após o loop, retorno a base final
    return dataset_unique_values.reset_index(drop=True)

# Função que separo o nome das variaveis por tipo (Categorica, Numerica e Datetime)
def columns_by_dtype(dataset):
    global columns_categorical, columns_numerical, columns_datetime
    columns_categorical = dataset.select_dtypes(include=['object','category']).columns.to_list()
    columns_numerical = dataset.select_dtypes(include=['number']).columns.to_list()
    columns_datetime = dataset.select_dtypes(include=['datetime', 'timedelta','datetimetz']).columns.to_list()

####################################################################################################
# Importação do dataset
####################################################################################################
# Importação
dataset = pd.read_csv('mental-health-in-tech-survey.csv')
print("Database shape after import: " + str(dataset.shape))

# Mostra informações básicas sobre a base, além de informações sobre as variaveis como: nome das variaveis, contagem frequencia de não nulos, tipo das variaveis
dataset.info()

# Renomeação de colunas (Opcional)
#dataset = dataset.rename(columns={"Var1_old":"Var1_new", "Var2_old":"Var2_new"}) 

#--------------------------------------------------------------------------------------------------
# Definição do tipo das variaveis
#--------------------------------------------------------------------------------------------------
# Definição da variavel output (ou dependente, resposta, target, etc).
column_output = ["treatment"]

# Conversão manual de tipo
dataset['Timestamp'] = dataset['Timestamp'].astype('datetime64')

# Separo o nome das variaveis por tipo (Categorica, Numerica e Datetime)    
columns_by_dtype(dataset)

####################################################################################################
# EDA - Explorative Data Analysis
####################################################################################################
#--------------------------------------------------------------------------------------------------
# Estatisticas Descritivas
#--------------------------------------------------------------------------------------------------
# Estatisticas descritiva basica das variaveis
dataset_raw_descriptive_statistics = descriptive_statistics(dataset = dataset)

# Valores unicos de todas variaveis
dataset_unique_values = unique_values(dataset = dataset, top_n = 10)

#--------------------------------------------------------------------------------------------------
# Visualizações
#--------------------------------------------------------------------------------------------------
# Histogramas
#TODO

# Densidade (Simetria (Skewness) e distribuição (Kurtosis))
#TODO

# Scatter plot
#TODO

# Box plot
#TODO

# Correlação
#TODO

####################################################################################################
# Data preprocessing
####################################################################################################
#--------------------------------------------------------------------------------------------------
# Remoção de instancias duplicadas (Opcional)
#--------------------------------------------------------------------------------------------------
print("Database shape before drop duplicates: " + str(dataset.shape))
dataset_temp = dataset.copy()
dataset_temp = dataset_temp.drop_duplicates()

# No final do tratamento, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset = dataset_temp
print("Database shape after drop duplicates: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)

#--------------------------------------------------------------------------------------------------
# Tratamento de valores missing - Remoção ou preenchimento
#--------------------------------------------------------------------------------------------------
# Faço uma copia da base original para fazer o tratamento. No final, se for tudo ok, substituo a original pela tratada
print("Database shape before dealing with missing: " + str(dataset.shape))
dataset_temp = dataset.copy()

# Pego o nome das variaveis numericas que possuem pelo menos 1 missing
columns_with_missing = dataset_descriptive_statistics[dataset_descriptive_statistics['Missing']>0]['Variable'].to_list()
print("Columns with missing values: " + str(columns_with_missing))

# Remoção de variaveis com uma porcentagem alta de missings
columns_many_missing = dataset_descriptive_statistics[dataset_descriptive_statistics['% missing']>85]['Variable'].to_list()
print("Columns with many missing values: " + str(columns_many_missing))
dataset_temp = dataset_temp.drop(columns=columns_many_missing)

# Remoção de instancias que contenham qualquer valor missing
#dataset_temp = dataset_temp.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) 

### Preenchimento em variaveis numéricas
# Pego o nome das variaveis numericas que possuem pelo menos 1 missing
columns_numerical_with_missing = dataset_descriptive_statistics[(dataset_descriptive_statistics['Missing']>0) & (dataset_descriptive_statistics['Variable'].isin(columns_numerical))]['Variable'].to_list()
print("Numerical columns with missing values: " + str(columns_numerical_with_missing))

# Preenchimento com um valor coringa
#dataset_temp[columns_numerical_with_missing] = dataset_temp[columns_numerical_with_missing].fillna(0)

# Preenchimento com estatistica basica (Opcional: agrupando pela variavel output)
#dataset_temp[columns_numerical_with_missing] = dataset_temp[columns_numerical_with_missing].fillna(dataset_temp[columns_numerical_with_missing].mean())
#dataset_temp[columns_numerical_with_missing] = dataset_temp[columns_numerical_with_missing].fillna(dataset_temp[columns_numerical_with_missing].median())
#dataset_temp[columns_numerical_with_missing] = dataset_temp[columns_numerical_with_missing].fillna(dataset_temp.groupby(column_output)[columns_numerical_with_missing].transform('mean'))
#dataset_temp[columns_numerical_with_missing] = dataset_temp[columns_numerical_with_missing].fillna(dataset_temp.groupby(column_output)[columns_numerical_with_missing].transform('median'))

# Preenchimento com regressão (Como tudo no sklearn, só funciona com as variaveis numericas. Se quiser utilizar as variaveis categoricas na regressao, terá que fazer encoding delas)
#imp = IterativeImputer(max_iter=10, random_state=0)
#imp.fit(dataset_temp[columns_numerical_with_missing])
#dataset_temp[columns_numerical_with_missing] = imp.transform(dataset_temp[columns_numerical_with_missing])

### Preenchimento em variaveis categoricas
# Pego o nome das variaveis categoricas que possuem pelo menos 1 missing
columns_categorical_with_missing = dataset_descriptive_statistics[(dataset_descriptive_statistics['Missing']>0) & (dataset_descriptive_statistics['Variable'].isin(columns_categorical))]['Variable'].to_list()
print("Categorical columns with missing values: " + str(columns_categorical_with_missing))

# Preenchimento manual para cada coluna
dict_categorical_fill= {'state': 'N/A'
                        , 'self_employed': '?'
                        , 'work_interfere': '?'
                        }
dataset_temp = dataset_temp.fillna(value=dict_categorical_fill)

# Preenchimento com o valor mais frequente
#dataset_temp[columns_categorical_with_missing] = dataset_temp[columns_categorical_with_missing].fillna(dataset_temp[columns_categorical_with_missing].mode().iloc[0])

# Preenchimento de todo valor missing com um valor coringa
#dataset_temp[columns_categorical_with_missing] = dataset_temp[columns_categorical_with_missing].fillna('?')

### Preenchimento em variaveis datetime
#TODO

# No final do tratamento, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset = dataset_temp
print("Database shape after dealing with missing: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)

#--------------------------------------------------------------------------------------------------
# Tratamento de outliers
#--------------------------------------------------------------------------------------------------
# Faço uma copia da base original para fazer o tratamento. No final, se for tudo ok, substituo a original pela tratada
print("Database shape before dealing with outliers: " + str(dataset.shape))
dataset_temp = dataset.copy()

### Em variaveis Numericas podemos usar o metodo interquartil ou zscore, e usar o tratamento de remover ou substituir pelo valor mais próximo aceitavel. Escolher somente um metodo e um tratamento.
dataset_temp.boxplot(column=columns_numerical)

## Interquartil
IQR = (dataset_temp.quantile(0.75) - dataset_temp.quantile(0.25))
upper_limit = (dataset_temp.quantile(0.75) + 1.5*IQR)
lower_limit = (dataset_temp.quantile(0.25) - 1.5*IQR)

# Removendo
#dataset_temp[columns_numerical] = dataset_temp[columns_numerical][((dataset_temp[columns_numerical]<upper_limit) & (dataset_temp[columns_numerical]>lower_limit)).all(axis=1)]

# Substituindo
dataset_temp[columns_numerical] = np.where((dataset_temp[columns_numerical]>upper_limit), upper_limit, dataset_temp[columns_numerical])
dataset_temp[columns_numerical] = np.where((dataset_temp[columns_numerical]<lower_limit), lower_limit, dataset_temp[columns_numerical])

## Z-Score
#media = dataset_temp[columns_numerical].mean()
#desv_pad = dataset_temp[columns_numerical].std()
#upper_limit = media + 3*desv_pad
#lower_limit = media - 3*desv_pad

# Removendo
#dataset_temp[columns_numerical][np.abs(dataset_temp[columns_numerical]-media) <= (3*desv_pad)]

# Substituindo
#dataset_temp[columns_numerical] = np.where((dataset_temp[columns_numerical]>upper_limit), upper_limit, dataset_temp[columns_numerical])
#dataset_temp[columns_numerical] = np.where((dataset_temp[columns_numerical]<lower_limit), lower_limit, dataset_temp[columns_numerical])

### Em variaveis Categoricas
# Arrumar manualmente valores
dataset_temp.loc[dataset_temp["Gender"].isin(["male", "M", "m"]), "Gender"] = "Male"
dataset_temp.loc[dataset_temp["Gender"].isin(["female", "F", "f"]), "Gender"] = "Female"
dataset_temp.loc[~dataset_temp["Gender"].isin(["Male", "Female"]), "Gender"] = "?"

# Tratar valores raros
#TODO

### Em variaveis Datetime
#TODO
 
# No final do tratamento, substituo a base real, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset = dataset_temp
print("Database shape after dealing with outliers: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)

#--------------------------------------------------------------------------------------------------
# Encoding e discretização
#--------------------------------------------------------------------------------------------------
# Faço uma copia da base original para fazer o tratamento. No final, se for tudo ok, substituo a original pela tratada
print("Database shape before dummys: " + str(dataset.shape))
dataset_temp = dataset.copy()

### Para variaveis categoricas
## Arrumar manualmente valores
dataset_temp.loc[dataset_temp["treatment"].isin(["Yes"]), "treatment"] = 1
dataset_temp.loc[dataset_temp["treatment"].isin(["No"]), "treatment"] = 0

dataset_temp.loc[dataset_temp["no_employees"].isin(["1-5"]), "no_employees"] = 1
dataset_temp.loc[dataset_temp["no_employees"].isin(["6-25"]), "no_employees"] = 2
dataset_temp.loc[dataset_temp["no_employees"].isin(["26-100"]), "no_employees"] = 3
dataset_temp.loc[dataset_temp["no_employees"].isin(["100-500"]), "no_employees"] = 4
dataset_temp.loc[dataset_temp["no_employees"].isin(["500-1000"]), "no_employees"] = 5
dataset_temp.loc[dataset_temp["no_employees"].isin(["More than 1000"]), "no_employees"] = 6

## Para variaveis com muitos valores distintos, podemos pegar os top_n ou top_% e acumular o resto em "outros", ou podemos converter os valores categoricos pelas suas frequencias
columns_categorical_many_uniques = dataset_descriptive_statistics[(dataset_descriptive_statistics["Variable"].isin(columns_categorical)) & (dataset_descriptive_statistics["unique"] > 10)]

# Transformando valores não muito frequentes em "outros"
dataset_unique_values_state = unique_values(dataset = dataset_temp[['state']], top_n = 10)
dataset_temp.loc[~dataset_temp["state"].isin(dataset_unique_values_state['Value']), "state"] = "Others"

# Transformando valores em suas frequencias
dataset_unique_values_Country = unique_values(dataset = dataset_temp[['Country']], top_n = 0)
rename_dict = dataset_unique_values_Country.set_index('Value').to_dict()['Count']
dataset_temp['Country'] = dataset_temp['Country'].replace(rename_dict)

## Função Labelencoder: transforma todos valores categoricos em inteiros de forma sequencial por variavel
# Criamos um dicionario para mapearmos os valores originais, para podermos usarmos futuramente para inverter o encoding e usar em novos dados
#dict_label_encoder = defaultdict(LabelEncoder)
#dataset_temp[columns_categorical] = dataset_temp[columns_categorical].apply(lambda x: dict_label_encoder[x.name].fit_transform(x))

# Para inverter e para novos dados
#dataset_temp[columns_categorical] = dataset_temp[columns_categorical].apply(lambda x: dict_label_encoder[x.name].inverse_transform(x))
#df_new = df_new.apply(lambda x: dict_label_encoder[x.name].transform(x))

## Função get_dummyes: transforma todos os valores categoricos em novas variaveis booleanas
dataset_temp = pd.get_dummies(data=dataset_temp, drop_first=True)

# Para novos dados, basta reindexarmos com as colunas da nossa base de treinamento e preencher vazios com 0
#df_new = pd.get_dummies(df_new)
#df_new.reindex(columns = dataset.columns, fill_value=0)


### Para variaveis numericas
## Discretizar em faixas de valores (uniform: Faixas tem range de valor parecido, quantile: Faixas tem a mesma quantidade de instancias, kmeans: Valores em cada faixa tem o mesmo centro mais proximo em um cluster de 1 dimensao )
#dataset_temp[columns_numerical] = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit_transform(dataset_temp[columns_numerical])

# No final do tratamento, substituo a base real, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset = dataset_temp
print("Database shape after dummys: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)

#--------------------------------------------------------------------------------------------------
# Normalização
#--------------------------------------------------------------------------------------------------
# Faço uma copia da base original para fazer o tratamento. No final, se for tudo ok, substituo a original pela tratada
print("Database shape before normalizing: " + str(dataset.shape))
dataset_temp = dataset[columns_numerical].values

### Os dois metodos mais comuns são Min-Max e a transformação normal
## Min-Max
min_max_scaler = MinMaxScaler()
dataset_temp = min_max_scaler.fit_transform(dataset_temp)

## Normal(Standard)
#standard_scaler = StandardScaler()
#dataset_temp = standard_scaler.fit_transform(dataset_temp) 

##Z-Score
#TODO

# No final do tratamento, substituo a base real, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset[columns_numerical] = pd.DataFrame(dataset_temp)
print("Database shape after normalizing: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)

#--------------------------------------------------------------------------------------------------
# Balanceamento de classes
#--------------------------------------------------------------------------------------------------
# Checamos o balanceamento das classes
dataset_unique_values = unique_values(dataset = dataset, top_n = 10)
dataset_unique_values[dataset_unique_values["Variable"] == column_output[0]]
pd.value_counts(dataset[column_output[0]]).plot.bar()

# Balanceando as classes
#TODO

#--------------------------------------------------------------------------------------------------
# Seleção de variaveis
#--------------------------------------------------------------------------------------------------
# TODO

#--------------------------------------------------------------------------------------------------
# PCA
#--------------------------------------------------------------------------------------------------
# Faço uma copia da base original para fazer o tratamento. No final, se for tudo ok, substituo a original pela tratada
print("Database shape before PCA: " + str(dataset.shape))
dataset_temp = dataset[columns_numerical].copy()

### Função PCA  
# PCA inicialmente sem um numero de componentes para avaliarmos o corte
pca_temp1 = PCA(n_components = None)
dataset_temp1 = pca_temp1.fit_transform(dataset_temp)

# Variancia de cada componente
pca_explained_variance_temp1 = pca_temp1.explained_variance_ratio_
pca_explained_variance_cumsum_temp1 =  pd.DataFrame(np.cumsum(pca_explained_variance_temp1))

pca_explained_variance_temp1[0:19].sum()

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(pca_explained_variance_cumsum_temp1)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

# Refazer o PCA após decidir quantos componentes serão utilizados
pca = PCA(n_components = 19)
dataset_temp = pca.fit_transform(dataset_temp)
pca_explained_variance = pca.explained_variance_ratio_
print("Total variancia explicada no PCA: " + str(pca_explained_variance.sum()))
pca_explained_variance_cumsum =  pd.DataFrame(np.cumsum(pca_explained_variance))

# Para novos dados
#dataset_new = pca.transform(dataset_new)

# No final do tratamento, substituo a base real, atualizo a base de estatisticas descritivas e o nome das variaveis por tipo
dataset_pca = dataset_temp
print("Database shape after PCA: " + str(dataset.shape))
dataset_descriptive_statistics = descriptive_statistics(dataset = dataset)
columns_by_dtype(dataset)



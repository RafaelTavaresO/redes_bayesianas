import pandas as pd

# Carrega o dataset original do Kaggle
df_completo = pd.read_csv('dataset_raw/weatherAUS.csv')

# Escolha das 7 variáveis para os nós
colunas_alvo = [
    'Location', 'Pressure3pm', 'WindDir3pm', 
    'Humidity3pm', 'Cloud3pm', 'RainToday', 'RainTomorrow'
]
df_rede = df_completo[colunas_alvo].copy()

# Limpeza - eliminando valores nulos do dataset
df_rede.dropna(inplace=True)

df_rede['Humidity3pm'] = pd.cut(
    df_rede['Humidity3pm'], 
    bins=[-1, 40, 70, 101], # Faixas: 0-40, 41-70, 71-100
    labels=['Baixa', 'Média', 'Alta']
)

df_rede['Pressure3pm'] = pd.cut(
    df_rede['Pressure3pm'], 
    bins=[0, 1010, 1020, 1050], 
    labels=['Baixa', 'Normal', 'Alta']
)

df_rede['Cloud3pm'] = pd.cut(
    df_rede['Cloud3pm'], 
    bins=[-1, 2, 5, 9], 
    labels=['Céu Limpo', 'Parcialmente Nublado', 'Nublado']
)

print("Amostra do dataset transformado para a Rede Bayesiana:\n")
print(df_rede.head())

df_rede.to_csv('dataset_chuva_preparado.csv', index=False)
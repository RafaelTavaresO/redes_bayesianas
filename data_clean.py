import pandas as pd

# Carrega o dataset original do Kaggle
df_completo = pd.read_csv('dataset_raw/weatherAUS.csv')

try:
    with open("exe.txt", "w") as file:
        file.write("Ola\n")
        file.write(",".join(str(v) for v in df_completo['Rainfall'].values))
except:
    print("\n")

# Escolha das 7 variáveis para os nós
colunas_alvo = [
    'Location', 'Pressure3pm', 'Rainfall', 
    'Humidity3pm', 'Cloud3pm', 'RainToday', 'RainTomorrow'
]
df_rede = df_completo[colunas_alvo].copy()

# Limpeza - eliminando valores nulos do dataset
df_rede['Rainfall'].fillna(0, inplace=True)
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

df_rede['Rainfall'] = pd.cut(
    df_rede['Rainfall'],
    bins=[-0.1, 1, 10, 50, 1000],
    labels=[
        'Sem Chuva',
        'Acumulo Leve',
        'Acumulo Médio',
        'Acumulo Alto'
    ]
)

print("Amostra do dataset transformado para a Rede Bayesiana:\n")
print(df_rede.head())

df_rede.to_csv('dataset_chuva_preparado.csv', index=False)

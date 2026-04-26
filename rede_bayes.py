from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np

def melhor_cenario_cpd(cpd, estado_alvo):
    # índice do estado alvo (ex: "Yes")
    estados = cpd.state_names[cpd.variable]
    idx_estado = estados.index(estado_alvo)

    # valores da linha correspondente (ex: P(RainTomorrow = Yes | ...)
    probs = cpd.values[idx_estado].flatten()

    # índice da maior probabilidade
    idx_max = np.argmax(probs)
    max_valor = probs[idx_max]

    # recuperar configuração dos pais
    evidencias = {}
    if cpd.get_evidence():
        evidencias_lista = cpd.get_evidence()[::-1]

        # calcular combinação de estados
        indices = np.unravel_index(idx_max, cpd.cardinality[1:])

        for var, i in zip(evidencias_lista, indices):
            evidencias[var] = cpd.state_names[var][i]

    return evidencias, max_valor

def gerar_cpd(df, var, pais):

    # Gera uma crosstab que correlaciona o nó Filho com os nós Pais
    tabela = pd.crosstab(
        [df[p] for p in pais] if pais else None,
        df[var],
        normalize='index'
    )

    # Armazena os estados que o nó Filho pode assumir
    estados_var = sorted(df[var].dropna().unique().tolist())

    # Armazena os estados que os nós Pais pode assumir
    estados_pais = [sorted(df[p].dropna().unique().tolist()) for p in pais]

    tabela = tabela[estados_var]
    valores = tabela.T.values

    # Cria um dicionário que conecta os nós Pais aos seus respectivos estados
    dicionario = dict(zip(pais, estados_pais))

    return TabularCPD(
        variable=var,
        variable_card=len(estados_var),
        values=valores,
        evidence=pais if pais else None,
        evidence_card=[len(dicionario[p]) for p in pais] if pais else None,
        state_names={var: estados_var, **dicionario }
    )

df_preparado = pd.read_csv('dataset_chuva_preparado.csv')

# Gerando CPD's utilizando o df, o nó Filho e uma lista de nós Pais
cpd_pressure = gerar_cpd(df_preparado, 'Pressure3pm', ['Location'])

cpd_windDir = gerar_cpd(df_preparado, 'WindDir3pm',['Location'])

cpd_humidity = gerar_cpd(df_preparado, 'Humidity3pm', ['Pressure3pm'])

cpd_cloud = gerar_cpd(df_preparado, 'Cloud3pm', ['Pressure3pm', 'Humidity3pm'])

cpd_rainTomorrow = gerar_cpd(df_preparado, 'RainTomorrow', ['Cloud3pm', 'Humidity3pm', 'WindDir3pm', 'RainToday'])

#print(cpd_rainTomorrow)

evidencias, max = melhor_cenario_cpd(cpd_pressure, 'Normal')

print(evidencias, max)
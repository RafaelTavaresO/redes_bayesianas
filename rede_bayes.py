from itertools import product
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np


def ranking_cenarios(cpd, estado_alvo):
    estados = cpd.state_names[cpd.variable]
    idx = estados.index(estado_alvo)

    probs = cpd.values[idx].flatten()
    pais = cpd.get_evidence()[::-1]

    resultados = []

    for i, p in enumerate(probs):
        indices = np.unravel_index(i, cpd.cardinality[1:])
        evidencias = {
            var: cpd.state_names[var][j]
            for var, j in zip(pais, indices)
        }
        resultados.append((evidencias, p))

    return sorted(resultados, key=lambda x: x[1], reverse=True)

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

def pior_cenario(cpd, estado_alvo):
    estados = cpd.state_names[cpd.variable]
    idx_yes = estados.index(estado_alvo)

    probs = cpd.values[idx_yes].flatten()

    idx_min = np.argmin(probs)
    min_val = probs[idx_min]

    pais = cpd.get_evidence()
    indices = np.unravel_index(idx_min, cpd.cardinality[1:])

    evidencias = {
        var: cpd.state_names[var][i]
        for var, i in zip(pais, indices)
    }

    return evidencias, min_val

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
    
    multi_index = pd.MultiIndex.from_tuples(
        list(product(*estados_pais)),
        names=pais)
    
    tabela = tabela.reindex(multi_index, fill_value=0)

    # garantir ordem correta das colunas
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

cpd_temp = gerar_cpd(df_preparado, 'Temp3pm', ['Location'])

cpd_humidity = gerar_cpd(df_preparado, 'Humidity3pm', ['Pressure3pm', 'Temp3pm'])

cpd_cloud = gerar_cpd(df_preparado, 'Cloud3pm', ['Pressure3pm', 'Humidity3pm'])

cpd_rainTomorrow = gerar_cpd(df_preparado, 'RainTomorrow', ['Rainfall', 'Cloud3pm', 'Humidity3pm'])

#print(cpd_rainTomorrow)

evidencia, max = melhor_cenario_cpd(cpd_rainTomorrow, 'Yes')

resultado = ranking_cenarios(cpd_rainTomorrow, 'Yes')

evidencia_2, min = pior_cenario(cpd_rainTomorrow, 'Yes')


print(evidencia, max)

print('\n')

i=0
for r in resultado[:5]:
    print("\n")
    print(r)

print('\n')

print(evidencia_2, min)

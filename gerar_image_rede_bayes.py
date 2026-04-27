from graphviz import Digraph
import matplotlib.pyplot as plt
import networkx as nx

def plotar_rede_matplotlib(edges):
    # Criando o grafo direcionado
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Layout (organização dos nós)
    pos = {
        'Location': (0, 2),
        'Pressure3pm': (-0.5, 1),
        'Temp3pm': (0.5, 1),
        'Humidity3pm': (0, -0.5),
        'Cloud3pm': (-0.5, -1.5),
        'Rainfall': (0.5, -1.5),
        'RainTomorrow': (0, -3)
    }   

    plt.figure(figsize=(8, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3250,
        font_size=8,
        arrows=True)

    plt.title("Rede Bayesiana - Modelo de Chuva")
    plt.tight_layout()

    # Salvar imagem
    plt.savefig("rede_bayesiana_matplotlib.png", dpi=300)

    # Mostrar
    plt.show()

def plotar_rede_graphviz(edges):
    dot = Digraph()

    for pai, filho in edges:
        dot.edge(pai, filho)

    dot.render("rede_bayesiana_graphviz", format="png", view=True)

edges = [
    ('Location', 'Pressure3pm'),
    ('Location', 'Temp3pm'),
    ('Temp3pm', 'Humidity3pm'),
    ('Pressure3pm', 'Humidity3pm'),
    ('Pressure3pm', 'Cloud3pm'),
    ('Humidity3pm', 'Cloud3pm'),
    ('Humidity3pm', 'RainTomorrow'),
    ('Cloud3pm', 'RainTomorrow'),
    ('Rainfall', 'RainTomorrow')
]

plotar_rede_graphviz(edges)
plotar_rede_matplotlib(edges)
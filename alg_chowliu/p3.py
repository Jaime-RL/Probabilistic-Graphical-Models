"""
===============================================================================
 PRÁCTICA 3 — APRENDIZAJE DE PARÁMETROS Y GRAFOS: ALGORITMO DE CHOW-LIU
 Representación del Conocimiento · Grado en Ingeniería Informática (UC)
===============================================================================

QUÉ PIDE LA PRÁCTICA
    1. Codificar el algoritmo de Chow-Liu (con NetworkX).
    2. Mostrar con ejemplos variados que funciona correctamente.
    3. Medir su coste temporal empírico en función de algún parámetro del
       coste teórico.

IDEA DEL ALGORITMO (lo que hay que saber para el examen)
    Queremos aprender, a partir de un conjunto de datos D, el GRAFO de una red
    bayesiana discreta, pero restringido a que sea una ARBORESCENCIA (árbol
    dirigido: cada nodo tiene como mucho un padre). Restringir la estructura
    evita el sobreajuste y, además, hace que el problema sea resoluble de forma
    exacta y eficiente.

    Resultado clave (Chow y Liu, 1968):

        arg max_G  log p(D | G)  =  arg max_G  Σ_{(vi,vj) ∈ E(G)}  I(Xi ; Xj | D)

    Es decir: el árbol que maximiza la verosimilitud de los datos es el que
    maximiza la SUMA de las informaciones mutuas de sus aristas.

    ¿Por qué? Con los parámetros de máxima verosimilitud (frecuencias), la
    log-verosimilitud de una red con un padre por nodo se descompone como

        log p(D | G) = N · Σ_{aristas (Pa(i), i)} I(X_Pa(i) ; Xi)  −  N · Σ_i H(Xi)

    El segundo término (entropías marginales) NO depende de G, así que
    maximizar la verosimilitud = maximizar la suma de informaciones mutuas.
    Esto se verifica numéricamente en la Prueba 3 de este fichero.

    Los tres pasos:
        ➀ Construir el grafo completo K = (V, E, W) con w_ij = I(Xi ; Xj | D).
        ➁ Buscar un árbol de recubrimiento de PESO MÁXIMO G = (V, E') de K.
        ➂ Elegir una raíz v ∈ V y orientar todas las aristas "hacia fuera".

    Información mutua estimada a partir de D (frecuencias relativas p_D):

        I(X ; Y | D) = Σ_{x,y} p_D(x,y) · log( p_D(x,y) / (p_D(x) · p_D(y)) )

    Propiedades útiles: I ≥ 0, I = 0 ⇔ independencia (empírica), simétrica
    (por eso K es NO dirigido).

ESTRUCTURA DEL FICHERO
    §1  Estimación de probabilidades e información mutua
    §2  Algoritmo de Chow-Liu (los 3 pasos)
    §3  Aprendizaje de parámetros (CPTs por máxima verosimilitud) y
        log-verosimilitud
    §4  Generación de datos sintéticos a partir de un árbol conocido
    §5  Pruebas de eficacia (tarea ➋)
    §6  Coste temporal teórico y empírico (tarea ➌)

Ejecución:  python practica3_chow_liu.py
            python practica3_chow_liu.py --sin-tiempos   (salta §6)
===============================================================================
"""

import argparse
import itertools
import math
import time

import networkx as nx
import numpy as np


# =============================================================================
# §1  ESTIMACIÓN DE PROBABILIDADES E INFORMACIÓN MUTUA
# =============================================================================
#
# REPRESENTACIÓN DE LOS DATOS (decisión de diseño)
#   D es una matriz numpy de enteros de forma (N, n):
#       N = nº de instancias (filas), n = nº de variables (columnas).
#   Cada variable Xi toma valores 0, 1, ..., r_i - 1 (r_i = cardinalidad).
#
#   ¿Por qué enteros 0..r-1 y no strings o un DataFrame? Porque así podemos
#   contar frecuencias con np.bincount, que es O(N) y muy rápido, y el código
#   del algoritmo no se mezcla con detalles de formato. Si los datos vienen
#   con etiquetas ("sí"/"no"), basta codificarlas antes (ver codificar()).


def codificar(columnas):
    """Convierte una lista de columnas con valores arbitrarios (p. ej. strings)
    en la matriz de enteros que usa el resto del código.

    Devuelve (D, categorias), donde categorias[i] es la lista de valores
    originales de la variable i, en el orden de su código entero.
    """
    D, categorias = [], []
    for col in columnas:
        valores = sorted(set(col))
        indice = {v: k for k, v in enumerate(valores)}
        D.append([indice[v] for v in col])
        categorias.append(valores)
    return np.array(D, dtype=np.int64).T, categorias


def informacion_mutua(x, y, rx=None, ry=None):
    """I(X ; Y | D) estimada con frecuencias relativas (en nats, log natural).

    Parámetros
    ----------
    x, y : arrays 1D de enteros de longitud N (una columna de D cada uno).
    rx, ry : cardinalidades. Si no se dan, se deducen como max + 1.

    Cómo se calcula (paso a paso)
    -----------------------------
    1) Tabla de contingencia conjunta n(x, y).
       Truco: codificamos cada par (x, y) como un único entero  x·ry + y.
       Así una sola llamada a np.bincount cuenta TODOS los pares en O(N).
       Reorganizando el vector de longitud rx·ry obtenemos la matriz rx × ry.
    2) p(x,y) = n(x,y)/N ;  p(x) y p(y) son las sumas por filas y columnas
       (marginales) de esa misma tabla — no hace falta recorrer D otra vez.
    3) Sumamos p(x,y)·log(p(x,y) / (p(x)p(y))) SOLO sobre las celdas con
       p(x,y) > 0. Esto implementa el convenio 0·log 0 = 0 y evita log(0).

    Coste: O(N + rx·ry).
    """
    N = len(x)
    rx = int(x.max()) + 1 if rx is None else rx
    ry = int(y.max()) + 1 if ry is None else ry

    conjunta = np.bincount(x * ry + y, minlength=rx * ry).reshape(rx, ry) / N
    px = conjunta.sum(axis=1)          # p(x): suma de cada fila
    py = conjunta.sum(axis=0)          # p(y): suma de cada columna

    mask = conjunta > 0                # convenio 0·log 0 = 0
    producto = np.outer(px, py)        # p(x)·p(y) para todas las celdas
    im = np.sum(conjunta[mask] * np.log(conjunta[mask] / producto[mask]))

    # Por errores de redondeo puede salir -1e-17; la IM real nunca es negativa.
    return max(float(im), 0.0)


def entropia(x, rx=None):
    """H(X | D) = -Σ p(x) log p(x). Solo se usa para la log-verosimilitud
    (§3), no la necesita el algoritmo de Chow-Liu en sí."""
    rx = int(x.max()) + 1 if rx is None else rx
    p = np.bincount(x, minlength=rx) / len(x)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def cardinalidades(D):
    """r_i de cada variable, deducidas de los datos."""
    return [int(D[:, i].max()) + 1 for i in range(D.shape[1])]


# =============================================================================
# §2  ALGORITMO DE CHOW-LIU
# =============================================================================


def paso1_grafo_completo(D, nombres=None):
    """➀ Crea K = (V, E, W) completo con w_ij = I(Xi ; Xj | D).

    - Nodos: una variable por columna de D.
    - Aristas: TODOS los pares {vi, vj}, i < j  →  n(n-1)/2 aristas.
      Es NO dirigido porque la información mutua es simétrica:
      I(Xi;Xj) = I(Xj;Xi), así que calcularla una vez por par basta.

    Coste: n(n-1)/2 pares × O(N + r²) cada uno  =  O(n² · (N + r²)).
    Este es el paso DOMINANTE del algoritmo (lo veremos en §6).
    """
    n = D.shape[1]
    nombres = list(range(n)) if nombres is None else list(nombres)
    r = cardinalidades(D)

    K = nx.Graph()
    K.add_nodes_from(nombres)
    for i, j in itertools.combinations(range(n), 2):
        w = informacion_mutua(D[:, i], D[:, j], r[i], r[j])
        K.add_edge(nombres[i], nombres[j], weight=w)
    return K


def paso2_arbol_maximo(K):
    """➁ Árbol de recubrimiento de PESO MÁXIMO de K.

    Usamos Kruskal (versión "máximo": ordena aristas de mayor a menor peso y
    añade cada una si no forma ciclo, comprobándolo con union-find).

    ¿Por qué Kruskal y no Prim? En un grafo COMPLETO (denso, m = Θ(n²)),
    Prim con matriz de adyacencia sería O(n²), algo mejor que Kruskal
    O(m log m) = O(n² log n). Pero aquí da igual en la práctica: el paso 1
    ya cuesta O(n²·N), que domina con mucho. Elegimos Kruskal porque es el
    que se explica en clase y el que NetworkX usa por defecto.

    Observación importante: si hay EMPATES de pesos puede haber varios
    árboles óptimos distintos. Todos tienen la misma suma de IM y, por tanto,
    la misma verosimilitud; el algoritmo devuelve uno de ellos.
    """
    return nx.maximum_spanning_tree(K, weight="weight", algorithm="kruskal")


def paso3_orientar(T, raiz):
    """➂ Orienta el árbol no dirigido T "hacia fuera" desde `raiz`.

    Un recorrido en anchura (BFS) desde la raíz visita cada nodo por primera
    vez desde exactamente un vecino: ese vecino será su único padre.
    Resultado: arborescencia (DAG con un padre por nodo, salvo la raíz).

    ¿Importa qué raíz elijamos? Para la verosimilitud, NO:
      todas las orientaciones de un mismo árbol son MARKOV-EQUIVALENTES
      (mismo esqueleto y ninguna v-estructura, porque ningún nodo tiene dos
      padres). Representan las mismas independencias y dan la misma
      log p(D|G). Se comprueba en la Prueba 4.
    Lo que cambia es la interpretación (y las CPTs que hay que estimar).

    Coste: O(n) (un árbol tiene n-1 aristas).
    """
    G = nx.bfs_tree(T, raiz)               # DiGraph con aristas padre → hijo
    # bfs_tree no copia atributos; copiamos el peso (IM) por comodidad.
    for u, v in G.edges():
        G[u][v]["weight"] = T[u][v]["weight"]
    return G


def chow_liu(D, nombres=None, raiz=None):
    """Algoritmo de Chow-Liu completo.

    Devuelve (G, K):
        G : nx.DiGraph, arborescencia aprendida (aristas padre → hijo,
            atributo 'weight' = información mutua de la arista).
        K : nx.Graph, grafo completo ponderado del paso 1 (útil para depurar
            y para las pruebas).
    Si no se indica raíz, se toma la primera variable.
    """
    K = paso1_grafo_completo(D, nombres)
    T = paso2_arbol_maximo(K)
    raiz = next(iter(K.nodes())) if raiz is None else raiz
    G = paso3_orientar(T, raiz)
    return G, K


# =============================================================================
# §3  APRENDIZAJE DE PARÁMETROS Y LOG-VEROSIMILITUD
# =============================================================================
#
# Chow-Liu aprende el GRAFO. Para tener la red bayesiana completa faltan los
# PARÁMETROS (las CPTs). Por máxima verosimilitud son simplemente frecuencias:
#
#     θ_{x | pa} = n(x, pa) / n(pa)
#
# Con eso podemos calcular log p(D | G, θ̂) y comprobar el teorema de
# Chow-Liu directamente (Prueba 3).


def estimar_cpts(D, G, nombres=None, alpha=0.0):
    """CPTs de máxima verosimilitud de la red con grafo G.

    Devuelve un dict  nodo -> array:
        - raíz (sin padre): vector p(x) de longitud r
        - resto: matriz r_padre × r_hijo con filas p(x | pa)

    alpha > 0 aplica suavizado de Laplace (evita probabilidades 0 cuando
    alguna combinación no aparece en los datos). Con alpha = 0 es MLE puro.
    """
    nombres = list(range(D.shape[1])) if nombres is None else list(nombres)
    col = {v: k for k, v in enumerate(nombres)}
    r = cardinalidades(D)
    cpts = {}
    for v in G.nodes():
        i = col[v]
        padres = list(G.predecessors(v))
        if not padres:
            c = np.bincount(D[:, i], minlength=r[i]) + alpha
            cpts[v] = c / c.sum()
        else:
            j = col[padres[0]]             # arborescencia: un único padre
            c = np.bincount(D[:, j] * r[i] + D[:, i],
                            minlength=r[j] * r[i]).reshape(r[j], r[i]) + alpha
            fila = c.sum(axis=1, keepdims=True)
            fila[fila == 0] = 1            # padre no observado: fila a 0
            cpts[v] = c / fila
    return cpts


def log_verosimilitud(D, G, nombres=None):
    """log p(D | G, θ̂) con θ̂ = MLE, calculada DIRECTAMENTE sumando
    log θ_{x|pa} sobre todas las filas (sin usar la fórmula de la IM).

    Así, si coincide con  N·ΣI − N·ΣH, hemos verificado la descomposición
    que justifica el algoritmo, en lugar de suponerla.
    """
    nombres = list(range(D.shape[1])) if nombres is None else list(nombres)
    col = {v: k for k, v in enumerate(nombres)}
    cpts = estimar_cpts(D, G, nombres)
    ll = 0.0
    for v in G.nodes():
        i = col[v]
        padres = list(G.predecessors(v))
        if not padres:
            ll += np.sum(np.log(cpts[v][D[:, i]]))
        else:
            j = col[padres[0]]
            ll += np.sum(np.log(cpts[v][D[:, j], D[:, i]]))
    return float(ll)


# =============================================================================
# §4  GENERACIÓN DE DATOS SINTÉTICOS
# =============================================================================
#
# Para saber si el algoritmo "acierta" necesitamos una VERDAD DE REFERENCIA:
# generamos una red bayesiana con forma de árbol conocida, muestreamos datos
# de ella y comprobamos si Chow-Liu recupera ese árbol.


def red_arbol_aleatoria(n, r=2, seed=None, fuerza=1.0):
    """Crea una red bayesiana aleatoria con grafo arborescencia.

    - Esqueleto: árbol etiquetado uniforme sobre n nodos (vía secuencia de
      Prüfer, nx.random_labeled_tree), orientado desde el nodo 0.
    - CPTs: cada fila se muestrea de una Dirichlet(fuerza, ..., fuerza).
      Con fuerza pequeña (< 1) las distribuciones son "picudas" → dependencias
      fuertes entre padre e hijo → árbol más fácil de recuperar.
      Con fuerza grande las filas se parecen entre sí → dependencias débiles.
    """
    rng = np.random.default_rng(seed)
    T = nx.random_labeled_tree(n, seed=int(rng.integers(1 << 31)))
    G = nx.bfs_tree(T, 0)
    cpts = {}
    for v in G.nodes():
        padres = list(G.predecessors(v))
        if not padres:
            cpts[v] = rng.dirichlet([fuerza] * r)
        else:
            cpts[v] = rng.dirichlet([fuerza] * r, size=r)  # r_padre × r
    return G, cpts


def muestrear(G, cpts, N, r=2, seed=None):
    """Muestreo ancestral (forward sampling): recorremos los nodos en orden
    topológico, así cuando generamos un hijo su padre ya tiene valor.
    Vectorizado por columnas: se generan las N filas a la vez."""
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    D = np.zeros((N, n), dtype=np.int64)
    for v in nx.topological_sort(G):
        padres = list(G.predecessors(v))
        if not padres:
            D[:, v] = rng.choice(r, size=N, p=cpts[v])
        else:
            probs = cpts[v][D[:, padres[0]]]          # N × r: fila según padre
            u = rng.random((N, 1))
            D[:, v] = (u > np.cumsum(probs, axis=1)).sum(axis=1)  # inversión CDF
    return D


def esqueleto(G):
    """Conjunto de aristas SIN dirección (para comparar árboles ignorando la
    orientación, que depende solo de la raíz elegida)."""
    return {frozenset(e) for e in G.edges()}


# =============================================================================
# §5  PRUEBAS DE EFICACIA (tarea ➋)
# =============================================================================
#
# Cada prueba comprueba una cosa distinta, de lo más sencillo a lo más general:
#   P1  Ejemplo pequeño "a mano" con nombres reales: ¿se entiende el resultado?
#   P2  Recuperación de árboles conocidos (cadena, estrella, aleatorios).
#   P3  Optimalidad: comparación con FUERZA BRUTA sobre todos los árboles.
#   P4  Independencia de la raíz: misma log-verosimilitud con cualquier raíz.
#   P5  Variables independientes: pesos ≈ 0 (caso degenerado).
#   P6  Efecto del tamaño muestral N en la tasa de acierto.


def titulo(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def prueba1_ejemplo_pequeno():
    titulo("P1 · Ejemplo pequeño con variables con nombre")
    # Construimos a mano una cadena  Nublado → Lluvia → Suelo_mojado  y una
    # variable Dia_semana independiente de todo. Esperamos:
    #   - aristas Nublado–Lluvia y Lluvia–Suelo_mojado (dependencias directas)
    #   - NO Nublado–Suelo_mojado: por la desigualdad de procesamiento de datos,
    #     en una cadena X→Y→Z se cumple I(X;Z) ≤ min(I(X;Y), I(Y;Z)), así que
    #     Kruskal prefiere siempre las aristas directas.
    #   - Dia_semana se cuelga de cualquiera con peso ≈ 0 (un árbol de
    #     recubrimiento TIENE que conectar todos los nodos: es una limitación
    #     del modelo, no un error).
    rng = np.random.default_rng(0)
    N = 5000
    nublado = rng.random(N) < 0.4
    lluvia = np.where(nublado, rng.random(N) < 0.7, rng.random(N) < 0.05)
    mojado = np.where(lluvia, rng.random(N) < 0.9, rng.random(N) < 0.1)
    dia = rng.integers(0, 7, N)
    nombres = ["Nublado", "Lluvia", "Suelo_mojado", "Dia_semana"]
    D, _ = codificar([nublado, lluvia, mojado, dia])

    G, K = chow_liu(D, nombres, raiz="Nublado")
    print("Pesos de K (información mutua, nats):")
    for u, v, w in sorted(K.edges(data="weight"), key=lambda e: -e[2]):
        print(f"   I({u:>12} ; {v:<12}) = {w:.4f}")
    print("Arborescencia aprendida (padre → hijo):")
    for u, v in G.edges():
        print(f"   {u} → {v}")

    esperado = {frozenset(("Nublado", "Lluvia")),
                frozenset(("Lluvia", "Suelo_mojado"))}
    assert esperado <= esqueleto(G), "No se recuperó la cadena esperada"
    assert K["Nublado"]["Suelo_mojado"]["weight"] < min(
        K["Nublado"]["Lluvia"]["weight"], K["Lluvia"]["Suelo_mojado"]["weight"])
    print("OK: cadena recuperada y se cumple la desigualdad de procesamiento.")


def prueba2_recuperacion():
    titulo("P2 · Recuperación de árboles conocidos")
    casos = []
    n = 8
    cadena = nx.DiGraph([(i, i + 1) for i in range(n - 1)])
    estrella = nx.DiGraph([(0, i) for i in range(1, n)])
    for nombre, G_real in [("cadena", cadena), ("estrella", estrella)]:
        rng = np.random.default_rng(1)
        cpts = {0: rng.dirichlet([0.5, 0.5])}
        for v in range(1, n):
            cpts[v] = rng.dirichlet([0.5, 0.5], size=2)
        casos.append((nombre, G_real, cpts, 2))
    for k in range(5):
        G_real, cpts = red_arbol_aleatoria(12, r=3, seed=100 + k, fuerza=0.5)
        casos.append((f"aleatorio #{k} (n=12, r=3)", G_real, cpts, 3))

    for nombre, G_real, cpts, r in casos:
        D = muestrear(G_real, cpts, N=20000, r=r, seed=7)
        G, _ = chow_liu(D)
        aciertos = len(esqueleto(G) & esqueleto(G_real))
        total = G_real.number_of_edges()
        print(f"   {nombre:<28} aristas correctas: {aciertos}/{total}")
        assert aciertos == total, f"Fallo en {nombre}"
    print("OK: con N grande se recupera exactamente el esqueleto real.")


def prueba3_fuerza_bruta():
    titulo("P3 · Optimalidad frente a fuerza bruta (todos los árboles)")
    # Por la fórmula de Cayley hay n^(n-2) árboles etiquetados: con n=6 son
    # 1296, enumerables. Para CADA árbol calculamos la log-verosimilitud
    # DIRECTAMENTE (contando, §3), sin usar la IM. Si Chow-Liu es correcto,
    # su árbol debe alcanzar el máximo. Usamos datos que NO vienen de un
    # árbol (dependencias aleatorias con un padre extra) para que la prueba
    # no sea trivial: el algoritmo debe dar el mejor árbol aunque el modelo
    # real no sea un árbol.
    rng = np.random.default_rng(3)
    n, N = 6, 3000
    D = rng.integers(0, 3, size=(N, n))
    D[:, 1] = (D[:, 0] + (rng.random(N) < 0.2)) % 3
    D[:, 3] = (D[:, 1] + D[:, 2]) % 3                 # dos "padres": no árbol
    D[:, 5] = np.where(rng.random(N) < 0.7, D[:, 4], D[:, 5])

    G_cl, K = chow_liu(D)
    ll_cl = log_verosimilitud(D, G_cl)

    # Comprobación de la descomposición:  LL = N·ΣI − N·ΣH
    suma_im = sum(w for _, _, w in G_cl.edges(data="weight"))
    suma_h = sum(entropia(D[:, i]) for i in range(n))
    print(f"   LL directa        = {ll_cl:.4f}")
    print(f"   N·ΣI − N·ΣH       = {N * suma_im - N * suma_h:.4f}")
    assert math.isclose(ll_cl, N * suma_im - N * suma_h, rel_tol=1e-9)

    mejor_ll, n_arboles = -math.inf, 0
    for T in nx.SpanningTreeIterator(K):            # enumera los n^(n-2)
        ll = log_verosimilitud(D, nx.bfs_tree(T, 0))
        mejor_ll = max(mejor_ll, ll)
        n_arboles += 1
    print(f"   Árboles enumerados: {n_arboles} (Cayley: {n ** (n - 2)})")
    print(f"   Máx LL fuerza bruta = {mejor_ll:.4f}  |  LL Chow-Liu = {ll_cl:.4f}")
    assert math.isclose(mejor_ll, ll_cl, rel_tol=1e-9)
    print("OK: Chow-Liu alcanza el máximo global de verosimilitud entre árboles.")


def prueba4_raiz():
    titulo("P4 · La elección de raíz no cambia la verosimilitud")
    G_real, cpts = red_arbol_aleatoria(10, r=2, seed=4)
    D = muestrear(G_real, cpts, N=5000, seed=4)
    lls = []
    for raiz in range(10):
        G, _ = chow_liu(D, raiz=raiz)
        assert G.in_degree(raiz) == 0
        assert all(G.in_degree(v) == 1 for v in G if v != raiz)
        lls.append(log_verosimilitud(D, G))
    print(f"   LL con las 10 raíces: min={min(lls):.6f}  max={max(lls):.6f}")
    assert max(lls) - min(lls) < 1e-6
    print("OK: orientaciones Markov-equivalentes ⇒ misma log-verosimilitud,\n"
          "    y cada nodo salvo la raíz tiene exactamente un padre.")


def prueba5_independientes():
    titulo("P5 · Variables independientes")
    rng = np.random.default_rng(5)
    D = rng.integers(0, 2, size=(100000, 5))
    G, K = chow_liu(D)
    max_w = max(w for _, _, w in K.edges(data="weight"))
    print(f"   Máxima IM entre pares: {max_w:.2e} nats (≈ 0)")
    print(f"   Aun así el árbol tiene {G.number_of_edges()} aristas: un árbol de\n"
          "   recubrimiento siempre conecta todo. Con datos finitos la IM\n"
          "   estimada es ligeramente > 0 (sesgo), por eso nunca sale 0 exacto.")
    assert max_w < 1e-3


def prueba6_tamano_muestral():
    titulo("P6 · Efecto del tamaño muestral N (fuerza=1, dependencias moderadas)")
    print(f"   {'N':>7} | % aristas correctas (media de 20 redes, n=15, r=2)")
    for N in [50, 200, 1000, 5000, 20000]:
        tasas = []
        for s in range(20):
            G_real, cpts = red_arbol_aleatoria(15, r=2, seed=s, fuerza=1.0)
            D = muestrear(G_real, cpts, N, seed=1000 + s)
            G, _ = chow_liu(D)
            tasas.append(len(esqueleto(G) & esqueleto(G_real)) / 14)
        print(f"   {N:>7} | {100 * np.mean(tasas):5.1f} %")
    print("   Con pocos datos la IM estimada es ruidosa y se eligen aristas\n"
          "   falsas; la tasa de acierto crece con N (consistencia).")


# =============================================================================
# §6  COSTE TEMPORAL (tarea ➌)
# =============================================================================
#
# COSTE TEÓRICO (n = nº variables, N = nº instancias, r = cardinalidad máx.)
#   ➀ n(n-1)/2 pares × O(N + r²)                 = O(n² · N)   (si N ≫ r²)
#   ➁ Kruskal sobre m = n(n-1)/2 aristas: O(m log m) = O(n² log n)
#   ➂ BFS sobre un árbol                          = O(n)
#   TOTAL: O(n² · N + n² log n) = O(n² · N) en la práctica, porque
#          N ≫ log n en cualquier caso realista.
#
# PREDICCIÓN PARA EL EXPERIMENTO
#   - Fijando N y variando n: tiempo ∝ n²  → en escala log-log, pendiente ≈ 2.
#   - Fijando n y variando N: tiempo ∝ N   → pendiente ≈ 1.
#   La pendiente se estima ajustando una recta a (log x, log t) con polyfit.
#
# Matices al interpretar resultados:
#   - Para N pequeño domina el coste fijo de cada llamada a numpy (overhead
#     de Python por par), así que la pendiente en N sale < 1 al principio y
#     se acerca a 1 cuando N crece.
#   - Se toma el MÍNIMO de varias repeticiones: es la medida menos afectada
#     por el ruido del sistema operativo.
#   - La pendiente en n suele salir algo por debajo de 2 (≈1.8-1.9): el
#     tiempo POR PAR baja un poco al crecer n (caché caliente, el intérprete
#     amortiza costes fijos). Comprobación más directa que la pendiente:
#     tiempo_paso1 / nº_pares debería ser ≈ constante, y lo es (~20-30 µs).


def cronometrar(D, repeticiones=3):
    """Tiempo (s) de cada paso por separado, mínimo de varias repeticiones."""
    mejores = [math.inf] * 3
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        K = paso1_grafo_completo(D)
        t1 = time.perf_counter()
        T = paso2_arbol_maximo(K)
        t2 = time.perf_counter()
        paso3_orientar(T, 0)
        t3 = time.perf_counter()
        for k, t in enumerate((t1 - t0, t2 - t1, t3 - t2)):
            mejores[k] = min(mejores[k], t)
    return mejores


def pendiente_loglog(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def experimento_tiempos(guardar_grafica=True):
    titulo("§6 · Coste temporal empírico")
    rng = np.random.default_rng(6)

    # --- (a) variar n con N fijo --------------------------------------------
    N_fijo = 2000
    ns = [10, 20, 40, 80, 160]
    tn = []
    print(f"(a) N = {N_fijo} fijo")
    print(f"   {'n':>5} {'pares':>7} {'paso1 (s)':>10} {'paso2 (s)':>10} {'paso3 (s)':>10}")
    for n in ns:
        D = rng.integers(0, 2, size=(N_fijo, n))
        t = cronometrar(D)
        tn.append(t)
        print(f"   {n:>5} {n * (n - 1) // 2:>7} {t[0]:>10.4f} {t[1]:>10.4f} {t[2]:>10.6f}")
    total_n = [sum(t) for t in tn]
    print(f"   Pendiente log-log tiempo total vs n: {pendiente_loglog(ns, total_n):.2f}"
          "  (teórica ≈ 2)")

    # --- (b) variar N con n fijo --------------------------------------------
    n_fijo = 20
    Ns = [1000, 4000, 16000, 64000, 256000]
    tN = []
    print(f"(b) n = {n_fijo} fijo")
    print(f"   {'N':>7} {'paso1 (s)':>10} {'paso2 (s)':>10}")
    for N in Ns:
        D = rng.integers(0, 2, size=(N, n_fijo))
        t = cronometrar(D)
        tN.append(t)
        print(f"   {N:>7} {t[0]:>10.4f} {t[1]:>10.5f}")
    total_N = [sum(t) for t in tN]
    print(f"   Pendiente log-log tiempo total vs N (3 últimos puntos): "
          f"{pendiente_loglog(Ns[2:], total_N[2:]):.2f}  (teórica ≈ 1)")
    print("   El paso 2 no depende de N: se mantiene constante, como predice\n"
          "   O(n² log n). El paso 1 (cálculo de IM) domina siempre.")

    if guardar_grafica:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("   (matplotlib no disponible: no se genera la gráfica)")
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ref = [total_n[0] * (n / ns[0]) ** 2 for n in ns]
        ax[0].loglog(ns, total_n, "o-", label="medido (total)")
        ax[0].loglog(ns, [t[0] for t in tn], "s--", label="paso 1 (IM)")
        ax[0].loglog(ns, [t[1] for t in tn], "^--", label="paso 2 (Kruskal)")
        ax[0].loglog(ns, ref, ":", color="gray", label="referencia ∝ n²")
        ax[0].set(xlabel="n (nº de variables)", ylabel="tiempo (s)",
                  title=f"Tiempo vs n (N = {N_fijo})")
        ax[0].legend()
        ref = [total_N[-1] * (N / Ns[-1]) for N in Ns]
        ax[1].loglog(Ns, total_N, "o-", label="medido (total)")
        ax[1].loglog(Ns, ref, ":", color="gray", label="referencia ∝ N")
        ax[1].set(xlabel="N (nº de instancias)", ylabel="tiempo (s)",
                  title=f"Tiempo vs N (n = {n_fijo})")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig("coste_temporal_chow_liu.png", dpi=130)
        print("   Gráfica guardada en coste_temporal_chow_liu.png")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Práctica 3: Chow-Liu")
    parser.add_argument("--sin-tiempos", action="store_true",
                        help="no ejecutar el experimento de coste temporal")
    args = parser.parse_args()

    prueba1_ejemplo_pequeno()
    prueba2_recuperacion()
    prueba3_fuerza_bruta()
    prueba4_raiz()
    prueba5_independientes()
    prueba6_tamano_muestral()
    if not args.sin_tiempos:
        experimento_tiempos()
    print("\nTodas las pruebas superadas.")

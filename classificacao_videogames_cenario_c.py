"""
Projeto AV2 - Inteligência Artificial Computacional
Disciplina: Inteligência Artificial Computacional
Objetivo: Avaliação de classificadores implementados manualmente.
Dataset: videogamesales.arff
Alvo: Genre
"""

import argparse
import csv
import math
import os
import random
import time
from collections import Counter

import numpy as np

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
ARQUIVO_PADRAO = "videogamesales.arff"
TARGET_NAME = "Genre"
SEED = 42
N_FOLDS_PADRAO = 5
K_VIZINHOS_PADRAO = 5
CHUNK_SIZE_KNN = 64
EPS = 1e-9

# ============================================================
# ETAPA 1: LEITURA MANUAL DO ARQUIVO ARFF
# ============================================================

def limpar_nome_atributo(nome):
    nome = nome.strip()
    if len(nome) >= 2 and ((nome[0] == "'" and nome[-1] == "'") or (nome[0] == '"' and nome[-1] == '"')):
        return nome[1:-1]
    return nome

def parse_attribute_line(line):
    rest = line.strip()[len("@attribute"):].strip()
    if rest.startswith("'"):
        fim = rest.find("'", 1)
        nome = rest[:fim + 1]
        tipo = rest[fim + 1:].strip()
    elif rest.startswith('"'):
        fim = rest.find('"', 1)
        nome = rest[:fim + 1]
        tipo = rest[fim + 1:].strip()
    else:
        partes = rest.split(None, 1)
        nome = partes[0]
        tipo = partes[1] if len(partes) > 1 else "string"

    nome = limpar_nome_atributo(nome)
    tipo_lower = tipo.strip().lower()

    if tipo_lower in ("numeric", "real", "integer"):
        tipo_simplificado = "numeric"
    elif tipo_lower == "string":
        tipo_simplificado = "string"
    elif tipo_lower.startswith("{"):
        tipo_simplificado = "nominal"
    else:
        tipo_simplificado = "string"

    return nome, tipo_simplificado, tipo

def carregar_arff(caminho):
    print(f"[ETAPA] Lendo o arquivo ARFF: {caminho}...")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    relation = None
    attributes = []
    data_lines = []
    lendo_dados = False

    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"): continue

            lower = line.lower()
            if lower.startswith("@relation"):
                partes = line.split(None, 1)
                relation = limpar_nome_atributo(partes[1]) if len(partes) > 1 else "sem_nome"
            elif lower.startswith("@attribute"):
                nome, tipo_simplificado, tipo_original = parse_attribute_line(line)
                attributes.append({"name": nome, "type": tipo_simplificado, "original_type": tipo_original})
            elif lower.startswith("@data"):
                lendo_dados = True
            elif lendo_dados:
                data_lines.append(line)

    data = []
    reader = csv.reader(data_lines, quotechar='"', skipinitialspace=True)
    for row in reader:
        if row: data.append([value.strip() for value in row])

    print(f"[ETAPA] Leitura concluída. {len(data)} instâncias e {len(attributes)} atributos carregados.")
    return relation, attributes, data

# ============================================================
# ETAPA 2: ANÁLISE E SEPARAÇÃO DE ATRIBUTOS
# ============================================================
# TEORIA: A IA não pode ver a resposta (alvo) misturada com as dicas (preditores).
# Precisamos separar o X (dados) do y (Gênero).

def is_missing(value):
    return value is None or str(value).strip() in ("?", "", "None", "nan", "NaN")

def safe_float(value):
    try: return float(value)
    except Exception: return np.nan

def contar_ausentes(data, col_idx):
    return sum(1 for row in data if is_missing(row[col_idx]))

def obter_coluna_por_nome(attributes, nome):
    for i, attr in enumerate(attributes):
        if attr["name"] == nome: return i
    raise ValueError(f"Coluna não encontrada: {nome}")

def separar_x_y(data, attributes):
    print("[ETAPA] Separando variáveis preditoras (X) da variável alvo (y)...")
    target_idx = obter_coluna_por_nome(attributes, TARGET_NAME)
    
    # TEORIA: Removendo viés e ruído.
    # Name: É inútil pois cada jogo tem um nome único (não gera padrão).
    # Global_Sales: É a soma exata das outras vendas regionais, o que deixaria o modelo viciado.
    nomes_removidos = {"Name", "Global_Sales"}
    
    feature_indices, feature_attrs = [], []
    for idx, attr in enumerate(attributes):
        if idx == target_idx or attr["name"] in nomes_removidos: continue
        feature_indices.append(idx)
        feature_attrs.append(attr)

    X_raw = [[row[i] for i in feature_indices] for row in data]
    y_raw = [row[target_idx] for row in data]

    return X_raw, y_raw, feature_attrs


def analisar_dataset(relation, attributes, data):
    print("[ETAPA] Analisando o dataset...")
    print(f"Relation: {relation}")
    print(f"Atributos: {len(attributes)}")
    print(f"Instâncias: {len(data)}")
    print("Atributos carregados:")
    for attr in attributes:
        print(f"  - {attr['name']} ({attr['type']})")
    print("[INFO] Análise inicial concluída.")


def mostrar_distribuicao_classes(y_raw):
    print("[ETAPA] Mostrando distribuição das classes...")
    from collections import Counter

    contagem = Counter(y_raw)
    total = len(y_raw)
    for classe, freq in sorted(contagem.items(), key=lambda item: (-item[1], item[0])):
        percentual = 100.0 * freq / total if total > 0 else 0.0
        print(f"  {classe}: {freq} ({percentual:.2f}%)")


def separar_x_y_cenario_c(data, attributes):
    return separar_x_y(data, attributes)


def preprocessar_fold_cenario_c(X_train_raw, X_test_raw, feature_attrs):
    return preprocessar_fold(X_train_raw, X_test_raw, feature_attrs)

def codificar_y(y_raw):
    # TEORIA (Label Encoding): Algoritmos matemáticos não entendem texto ("Action", "RPG"). 
    # Esta função converte textos em números inteiros (0, 1, 2...).
    print("[ETAPA] Realizando Label Encoding da variável alvo (Genre)...")
    classes = sorted(set(y_raw))
    class_to_int = {classe: i for i, classe in enumerate(classes)}
    int_to_class = {i: classe for classe, i in class_to_int.items()}
    y = np.array([class_to_int[v] for v in y_raw], dtype=int)
    return y, class_to_int, int_to_class

# ============================================================
# ETAPA 3: PRÉ-PROCESSAMENTO E FEATURES DERIVADAS
# ============================================================

def ajustar_preprocessador_base(X_train_raw, feature_attrs):
    preprocessador = []
    for j, attr in enumerate(feature_attrs):
        coluna = [row[j] for row in X_train_raw]
        if attr["type"] == "numeric":
            valores = np.array([safe_float(v) for v in coluna], dtype=float)
            media = np.nanmean(valores)
            preprocessador.append({"name": attr["name"], "type": "numeric", "mean_fill": float(media if not np.isnan(media) else 0.0)})
        else:
            categorias = sorted(set(v for v in coluna if not is_missing(v)))
            mapping = {cat: idx for idx, cat in enumerate(categorias)}
            preprocessador.append({"name": attr["name"], "type": "categorical", "mapping": mapping, "unknown_value": -1.0})
    return preprocessador

def transformar_base(X_raw, preprocessador):
    X = np.zeros((len(X_raw), len(preprocessador)), dtype=float)
    for i, row in enumerate(X_raw):
        for j, config in enumerate(preprocessador):
            valor = row[j]
            if config["type"] == "numeric":
                v = safe_float(valor)
                X[i, j] = config["mean_fill"] if (is_missing(valor) or np.isnan(v)) else v
            else:
                X[i, j] = config["unknown_value"] if is_missing(valor) else float(config["mapping"].get(valor, config["unknown_value"]))
    return X

def indice_feature(feature_names, nome):
    try: return feature_names.index(nome)
    except ValueError: raise ValueError(f"Feature '{nome}' não encontrada.")

def ajustar_parametros_features_derivadas(X_train_base, feature_names):
    idx_year = indice_feature(feature_names, "Year")
    years = X_train_base[:, idx_year]
    years_validos = years[np.isfinite(years)]
    max_year_ref = float(np.max(years_validos)) if len(years_validos) > 0 else 0.0
    return {"max_year_ref": max_year_ref}

def criar_features_derivadas(X_base, feature_names, parametros):
    # TEORIA (Feature Engineering): Como tiramos duas colunas, o dataset ficou pobre.
    # Criamos cálculos (ex: Porcentagem de venda no Japão vs Europa, Década de lançamento)
    # para dar mais "pistas" inteligentes ao modelo sem trapacear.
    idx_rank, idx_year = indice_feature(feature_names, "Rank"), indice_feature(feature_names, "Year")
    idx_na, idx_eu = indice_feature(feature_names, "NA_Sales"), indice_feature(feature_names, "EU_Sales")
    idx_jp, idx_other = indice_feature(feature_names, "JP_Sales"), indice_feature(feature_names, "Other_Sales")

    rank, year = X_base[:, idx_rank], X_base[:, idx_year]
    na, eu, jp, other = X_base[:, idx_na], X_base[:, idx_eu], X_base[:, idx_jp], X_base[:, idx_other]

    total_sales = na + eu + jp + other
    total_safe = np.where(total_sales > EPS, total_sales, EPS)

    na_share, eu_share, jp_share, other_share = na / total_safe, eu / total_safe, jp / total_safe, other / total_safe
    log_total_sales = np.log1p(np.maximum(total_sales, 0.0))

    rank_inverse = 1.0 / np.where(rank > EPS, rank, EPS)
    rank_log = np.log1p(np.maximum(rank, 0.0))

    game_age = np.maximum(parametros["max_year_ref"] - year, 0.0)
    decade = np.floor(year / 10.0) * 10.0

    market_count = (na > 0).astype(float) + (eu > 0).astype(float) + (jp > 0).astype(float) + (other > 0).astype(float)
    
    shares = np.vstack([na_share, eu_share, jp_share, other_share]).T
    regional_diversity = -np.sum(shares * np.log(shares + EPS), axis=1)

    na_eu_diff = na - eu
    non_jp_sales = na + eu + other
    jp_nonjp_ratio = jp / np.where(non_jp_sales > EPS, non_jp_sales, EPS)

    derivadas = np.column_stack([
        total_sales, na_share, eu_share, jp_share, other_share, log_total_sales,
        rank_inverse, rank_log, game_age, decade, market_count, regional_diversity,
        na_eu_diff, jp_nonjp_ratio,
    ])

    nomes_derivados = ["Total_Sales_Manual", "NA_Share", "EU_Share", "JP_Share", "Other_Share", 
                       "Log_Total_Sales", "Rank_Inverse", "Rank_Log", "Game_Age", "Decade", 
                       "Market_Count", "Regional_Diversity", "NA_EU_Diff", "JP_NonJP_Ratio"]

    return np.hstack([X_base, derivadas]), feature_names + nomes_derivados

def normalizar_treino_teste(X_train, X_test):
    # TEORIA (Z-Score): Garante que vendas de "1.000.000" não tenham mais peso 
    # matemático que o ano de lançamento "2015". Tudo fica na mesma escala.
    # OBS CRÍTICA: Calculamos a média APENAS no treino, para não "espiar" os dados de teste.
    media = X_train.mean(axis=0)
    desvio = X_train.std(axis=0)
    desvio[desvio < EPS] = 1.0
    return (X_train - media) / desvio, (X_test - media) / desvio

def preprocessar_fold(X_train_raw, X_test_raw, feature_attrs):
    preprocessador = ajustar_preprocessador_base(X_train_raw, feature_attrs)
    X_train_base = transformar_base(X_train_raw, preprocessador)
    X_test_base = transformar_base(X_test_raw, preprocessador)

    feature_names_base = [attr["name"] for attr in feature_attrs]
    parametros_derivadas = ajustar_parametros_features_derivadas(X_train_base, feature_names_base)

    X_train_der, feature_names_final = criar_features_derivadas(X_train_base, feature_names_base, parametros_derivadas)
    X_test_der, _ = criar_features_derivadas(X_test_base, feature_names_base, parametros_derivadas)

    X_train_norm, X_test_norm = normalizar_treino_teste(X_train_der, X_test_der)
    return X_train_norm, X_test_norm, feature_names_final

# ============================================================
# ETAPA 4: ALGORITMOS DE CLASSIFICAÇÃO
# ============================================================
# TEORIA GERAL: Modelos tentam descobrir a classe de um dado novo (Teste)
# com base nos padrões que aprenderam com os dados velhos (Treino).

def dividir_k_folds(n_amostras, n_folds=5, seed=42):
    indices = list(range(n_amostras))
    random.Random(seed).shuffle(indices)
    return [fold.tolist() for fold in np.array_split(np.array(indices, dtype=int), n_folds)]

def distancia_euclidiana_matriz(X_chunk, X_train):
    # TEORIA: Calcula a distância geométrica em linha reta (Hipotenusa / Pitágoras).
    # Analisa a diferença global direta entre dois jogos no "espaço matemático".
    chunk_norm = np.sum(X_chunk * X_chunk, axis=1)[:, None]
    train_norm = np.sum(X_train * X_train, axis=1)[None, :]
    return np.maximum(chunk_norm + train_norm - 2.0 * (X_chunk @ X_train.T), 0.0)

def distancia_manhattan_matriz(X_chunk, X_train):
    # TEORIA: Calcula a distância somando o deslocamento de cada eixo separadamente 
    # (como andar em quarteirões numa cidade). Sofre menos impacto de Outliers (valores absurdos).
    return np.sum(np.abs(X_chunk[:, None, :] - X_train[None, :, :]), axis=2)

def knn_predict_batch(X_train, y_train, X_test, k=5, metrica="euclidiana", chunk_size=64):
    # kNN (K-Nearest Neighbors): Algoritmo de "votação de vizinhos".
    # Pega um jogo novo, encontra os 'k' jogos de treino mais parecidos com ele, e a classe majoritária vence.
    y_train = np.asarray(y_train, dtype=int)
    n_classes = int(np.max(y_train)) + 1
    predicoes = []

    for inicio in range(0, len(X_test), chunk_size):
        fim = min(inicio + chunk_size, len(X_test))
        X_chunk = X_test[inicio:fim]
        distancias = distancia_euclidiana_matriz(X_chunk, X_train) if metrica == "euclidiana" else distancia_manhattan_matriz(X_chunk, X_train)
        
        indices_k = np.argpartition(distancias, kth=k - 1, axis=1)[:, :k]
        vizinhos_classes = y_train[indices_k]
        for linha in vizinhos_classes:
            predicoes.append(int(np.argmax(np.bincount(linha, minlength=n_classes))))

    return np.array(predicoes, dtype=int)

def treinar_bayes_univariado(X_train, y_train, n_classes):
    # BAYES UNIVARIADO (Ingênuo): Calcula a média e desvio de cada atributo isoladamente.
    # Analisa os dados supondo que "Vendas no Japão" e "Vendas nos EUA" não tem relação NENHUMA entre si.
    modelo = {}
    for classe in range(n_classes):
        Xc = X_train[y_train == classe]
        if len(Xc) == 0: continue
        desvio = Xc.std(axis=0)
        desvio[desvio < EPS] = EPS
        modelo[classe] = {"media": Xc.mean(axis=0), "desvio": desvio, "prior": len(Xc) / len(y_train)}
    return modelo

def prever_bayes_univariado_batch(modelo, X_test):
    classes = sorted(modelo.keys())
    log_probs = np.zeros((len(X_test), len(classes)), dtype=float)

    for idx, classe in enumerate(classes):
        var = modelo[classe]["desvio"] ** 2
        log_likelihood = -0.5 * np.sum(np.log(2 * math.pi * var) + ((X_test - modelo[classe]["media"]) ** 2) / var, axis=1)
        log_probs[:, idx] = math.log(modelo[classe]["prior"] + EPS) + log_likelihood

    return np.array([classes[i] for i in np.argmax(log_probs, axis=1)], dtype=int)

def treinar_bayes_multivariado(X_train, y_train, n_classes):
    # BAYES MULTIVARIADO: Usa a Matriz de Covariância.
    # Analisa a RELAÇÃO entre as variáveis. Ele entende, por exemplo, que "Vender muito na Europa" 
    # E "Vender pouco no Japão" formam um padrão específico (correlação).
    modelo = {}
    d = X_train.shape[1]
    for classe in range(n_classes):
        Xc = X_train[y_train == classe]
        if len(Xc) == 0: continue
        
        cov = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.eye(d)
        if cov.ndim == 0: cov = np.array([[float(cov)]])
        
        cov += 1e-6 * np.eye(d)
        inv_cov = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        
        if sign <= 0:
            cov += 1e-4 * np.eye(d)
            inv_cov = np.linalg.pinv(cov)
            sign, logdet = np.linalg.slogdet(cov)

        modelo[classe] = {"media": Xc.mean(axis=0), "inv_cov": inv_cov, "logdet": float(logdet), "prior": len(Xc) / len(y_train), "d": d}
    return modelo

def prever_bayes_multivariado_batch(modelo, X_test):
    classes = sorted(modelo.keys())
    log_probs = np.zeros((len(X_test), len(classes)), dtype=float)

    for idx, classe in enumerate(classes):
        mod = modelo[classe]
        diff = X_test - mod["media"]
        quad = np.sum((diff @ mod["inv_cov"]) * diff, axis=1)
        log_likelihood = -0.5 * (mod["d"] * math.log(2 * math.pi) + mod["logdet"] + quad)
        log_probs[:, idx] = math.log(mod["prior"] + EPS) + log_likelihood

    return np.array([classes[i] for i in np.argmax(log_probs, axis=1)], dtype=int)

# ============================================================
# ETAPA 5: CÁLCULO DE MÉTRICAS (MANUAL)
# ============================================================

def matriz_confusao(y_true, y_pred, n_classes):
    # TEORIA: Tabela que cruza "O que era de verdade" vs "O que o modelo chutou".
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_true, y_pred):
        matriz[int(real), int(pred)] += 1
    return matriz

def calcular_metricas(y_true, y_pred, n_classes):
    cm = matriz_confusao(y_true, y_pred, n_classes)
    total = np.sum(cm)
    
    # ACURÁCIA (Accuracy): De todos os jogos testados, quantos % o modelo acertou o gênero?
    accuracy = np.trace(cm) / total if total > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    for classe in range(n_classes):
        tp = cm[classe, classe]
        fp = np.sum(cm[:, classe]) - tp
        fn = np.sum(cm[classe, :]) - tp

        # PRECISÃO (Precision): De todos que o modelo CHUTOU ser RPG, quantos realmente eram RPG? 
        # (Avalia se o modelo tá mentindo/chutando muito).
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # RECALL (Sensibilidade): De todos os RPGs REAIS no dataset, quantos o modelo conseguiu ENCONTRAR?
        # (Avalia se o modelo é cego para certas classes).
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-SCORE: Média harmônica entre Precisão e Recall. É a nota mais justa e importante
        # quando temos um dataset desbalanceado (muitos jogos de Ação, poucos de Puzzle).
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {"accuracy": float(accuracy), "precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)), "f1": float(np.mean(f1s))}

# ============================================================
# ETAPA 6: LOOP DE VALIDAÇÃO CRUZADA E EXECUÇÃO GERAL
# ============================================================

def limitar_treino_knn(X_train, y_train, max_treino_knn, seed):
    if max_treino_knn is None or len(X_train) <= max_treino_knn:
        return X_train, y_train
    indices = np.random.default_rng(seed).choice(len(X_train), size=max_treino_knn, replace=False)
    return X_train[indices], y_train[indices]

def adicionar_resultado(resultados, nome_modelo, metricas, tempo_treino, tempo_teste):
    resultados[nome_modelo]["accuracy"].append(metricas["accuracy"])
    resultados[nome_modelo]["precision"].append(metricas["precision"])
    resultados[nome_modelo]["recall"].append(metricas["recall"])
    resultados[nome_modelo]["f1"].append(metricas["f1"])
    resultados[nome_modelo]["train_time"].append(tempo_treino)
    resultados[nome_modelo]["test_time"].append(tempo_teste)

def executar_validacao_cruzada(X_raw, y, feature_attrs, n_classes, n_folds=5, k_vizinhos=5, max_treino_knn=None, seed=42):
    print(f"\n[ETAPA] Iniciando Validação Cruzada k-Fold (k={n_folds})...")
    modelos = ["kNN Euclidiana", "kNN Manhattan", "Bayesiano Univariado", "Bayesiano Multivariado"]
    resultados = {nome: {m: [] for m in ["accuracy", "precision", "recall", "f1", "train_time", "test_time"]} for nome in modelos}

    folds = dividir_k_folds(len(X_raw), n_folds=n_folds, seed=seed)
    todos_indices = set(range(len(X_raw)))

    for fold_idx, test_indices in enumerate(folds, start=1):
        test_set = set(test_indices)
        train_indices = list(todos_indices - test_set)

        X_train_raw, X_test_raw = [X_raw[i] for i in train_indices], [X_raw[i] for i in test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        print(f"\n  ── Processando Fold {fold_idx}/{n_folds} ────────────────────────────────")
        print("    ↳ Criando features derivadas e Normalizando os dados localmente...")
        X_train, X_test, _ = preprocessar_fold(X_train_raw, X_test_raw, feature_attrs)

        # kNN Euclidiana
        print("    ↳ Treinando e Testando: kNN Euclidiana...")
        t0 = time.perf_counter()
        X_knn, y_knn = limitar_treino_knn(X_train, y_train, max_treino_knn, seed + fold_idx)
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = knn_predict_batch(X_knn, y_knn, X_test, k=k_vizinhos, metrica="euclidiana", chunk_size=CHUNK_SIZE_KNN)
        t_teste = time.perf_counter() - t0
        adicionar_resultado(resultados, "kNN Euclidiana", calcular_metricas(y_test, pred, n_classes), t_treino, t_teste)

        # kNN Manhattan
        print("    ↳ Treinando e Testando: kNN Manhattan...")
        t0 = time.perf_counter()
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = knn_predict_batch(X_knn, y_knn, X_test, k=k_vizinhos, metrica="manhattan", chunk_size=CHUNK_SIZE_KNN)
        t_teste = time.perf_counter() - t0
        adicionar_resultado(resultados, "kNN Manhattan", calcular_metricas(y_test, pred, n_classes), t_treino, t_teste)

        # Bayes Univariado
        print("    ↳ Treinando e Testando: Bayesiano Univariado...")
        t0 = time.perf_counter()
        modelo_uni = treinar_bayes_univariado(X_train, y_train, n_classes)
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = prever_bayes_univariado_batch(modelo_uni, X_test)
        t_teste = time.perf_counter() - t0
        adicionar_resultado(resultados, "Bayesiano Univariado", calcular_metricas(y_test, pred, n_classes), t_treino, t_teste)

        # Bayes Multivariado
        print("    ↳ Treinando e Testando: Bayesiano Multivariado...")
        t0 = time.perf_counter()
        modelo_multi = treinar_bayes_multivariado(X_train, y_train, n_classes)
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = prever_bayes_multivariado_batch(modelo_multi, X_test)
        t_teste = time.perf_counter() - t0
        adicionar_resultado(resultados, "Bayesiano Multivariado", calcular_metricas(y_test, pred, n_classes), t_treino, t_teste)

    return resultados

def media_desvio(valores):
    return float(np.mean(valores)), float(np.std(valores))

def resumir_resultados(resultados):
    resumo = {}
    for modelo, metricas in resultados.items():
        resumo[modelo] = {}
        for metrica, valores in metricas.items():
            media, desvio = media_desvio(valores)
            resumo[modelo][metrica] = {"mean": media, "std": desvio}
    return resumo

def formatar_media_desvio(resumo_modelo, chave, casas=4):
    return f"{resumo_modelo[chave]['mean']:.{casas}f} ± {resumo_modelo[chave]['std']:.{casas}f}"

def imprimir_tabela_resultados(resumo):
    print("\n" + "=" * 124)
    print("TABELA DE RESULTADOS FINAIS - CLASSIFICAÇÃO")
    print("=" * 124)
    print(f"{'Classificador':<28} {'Acurácia':>17} {'Precisão':>17} {'Recall':>17} {'F1-Score':>17} {'T.Treino(s)':>17} {'T.Teste(s)':>17}")
    print("─" * 124)
    for modelo in resumo:
        r = resumo[modelo]
        print(f"{modelo:<28} {formatar_media_desvio(r, 'accuracy'):>17} {formatar_media_desvio(r, 'precision'):>17} "
              f"{formatar_media_desvio(r, 'recall'):>17} {formatar_media_desvio(r, 'f1'):>17} "
              f"{formatar_media_desvio(r, 'train_time'):>17} {formatar_media_desvio(r, 'test_time'):>17}")
    print("=" * 124)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", default=ARQUIVO_PADRAO, help="Caminho para o arquivo videogamesales.arff")
    parser.add_argument("--quick", action="store_true", help="Modo rápido para testes iniciais")
    parser.add_argument("--max-treino-knn", type=int, default=None, help="Limita o treino do kNN para desempenho.")
    args = parser.parse_args()

    n_folds = 2 if args.quick else N_FOLDS_PADRAO
    max_treino_knn = 1000 if args.quick else args.max_treino_knn
    inicio_total = time.perf_counter()

    print("=" * 86)
    print("INÍCIO DO PIPELINE DE APRENDIZADO DE MÁQUINA")
    print("=" * 86)

    relation, attributes, data = carregar_arff(args.arquivo)
    X_raw, y_raw, feature_attrs = separar_x_y(data, attributes)
    y, class_to_int, int_to_class = codificar_y(y_raw)
    n_classes = len(class_to_int)

    resultados = executar_validacao_cruzada(
        X_raw, y, feature_attrs, n_classes, n_folds=n_folds, k_vizinhos=K_VIZINHOS_PADRAO, max_treino_knn=max_treino_knn, seed=SEED
    )

    print("\n[ETAPA] Gerando Resumo dos Resultados...")
    resumo = resumir_resultados(resultados)
    imprimir_tabela_resultados(resumo)
    print(f"\n[INFO] Pipeline finalizado! Tempo total de execução: {time.perf_counter() - inicio_total:.1f}s")

if __name__ == "__main__":
    main()
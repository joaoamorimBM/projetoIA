"""
Projeto AV3 - Inteligência Artificial Computacional
Disciplina: Inteligência Artificial Computacional - UNIFOR

Objetivo:
    Continuação da AV2 para a tarefa de REGRESSÃO.
    Este arquivo implementa manualmente uma MLP (Multi Layer Perceptron)
    para regressão, sem pandas e sem scikit-learn.

Dataset esperado:
    datasetRegressao.arff

O que este código faz:
    1. Lê manualmente o arquivo ARFF.
    2. Separa X (preditores) e y (alvo numérico).
    3. Trata valores ausentes.
    4. Codifica atributos categóricos manualmente.
    5. Normaliza X e y usando apenas o treino em cada fold.
    6. Implementa uma MLP manual com forward propagation e backpropagation.
    7. Testa diferentes topologias, ativações, learning rates e épocas.
    8. Calcula MSE, RMSE, MAE, R² e R² ajustado.
    9. Mede tempo de treino e tempo de teste.
    10. Salva relatório em resultados_av3/resultados_mlp.txt.

Observação importante:
    A MLP usa saída linear, pois estamos resolvendo REGRESSÃO.
    Em regressão, a saída não é uma classe; é um valor numérico contínuo.
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
ARQUIVO_PADRAO = "datasetRegressao.arff"
SEED = 42
N_FOLDS_PADRAO = 5
EPS = 1e-9
RESULTADOS_DIR = "resultados_av3"
RESULTADOS_ARQUIVO = os.path.join(RESULTADOS_DIR, "resultados_mlp.txt")

# Limite usado apenas para evitar explosão numérica durante o treino.
CLIP_GRADIENTE = 5.0


# ============================================================
# ETAPA 1: LEITURA MANUAL DO ARQUIVO ARFF
# ============================================================

def limpar_nome_atributo(nome):
    """Remove aspas simples ou duplas do nome do atributo."""
    nome = nome.strip()
    if len(nome) >= 2 and ((nome[0] == "'" and nome[-1] == "'") or (nome[0] == '"' and nome[-1] == '"')):
        return nome[1:-1]
    return nome


def parse_attribute_line(line):
    """
    Interpreta uma linha @attribute do ARFF.

    Exemplos:
        @attribute CPU numeric
        @attribute GameSetting {low,med,high,max}
        @attribute 'Nome da coluna' real
    """
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
    """
    Lê manualmente um arquivo ARFF.

    Retorna:
        relation: nome da relação/dataset
        attributes: lista de atributos com nome e tipo
        data: linhas brutas dos dados, ainda como texto
    """
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
            if not line or line.startswith("%"):
                continue

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
    # O csv.reader lida melhor com vírgulas e aspas do que split(',').
    reader = csv.reader(data_lines, quotechar='"', skipinitialspace=True)
    for row in reader:
        if row:
            data.append([value.strip() for value in row])

    print(f"[ETAPA] Leitura concluída. {len(data)} instâncias e {len(attributes)} atributos carregados.")
    return relation, attributes, data


# ============================================================
# ETAPA 2: ANÁLISE E PREPARAÇÃO DO DATASET
# ============================================================

def is_missing(value):
    """Verifica se um valor representa ausência no ARFF."""
    return value is None or str(value).strip() in ("?", "", "None", "nan", "NaN")


def safe_float(value):
    """Converte valor para float; se falhar, retorna np.nan."""
    try:
        return float(str(value).replace("'", "").replace('"', ""))
    except Exception:
        return np.nan


def contar_ausentes(data, col_idx):
    return sum(1 for row in data if col_idx >= len(row) or is_missing(row[col_idx]))


def analisar_dataset(relation, attributes, data):
    """Mostra informações iniciais do dataset."""
    print("[ETAPA] Analisando o dataset...")
    print(f"Relation: {relation}")
    print(f"Atributos: {len(attributes)}")
    print(f"Instâncias: {len(data)}")
    print("Atributos carregados:")
    for i, attr in enumerate(attributes):
        ausentes = contar_ausentes(data, i)
        print(f"  {i:02d}. {attr['name']} ({attr['type']}) | ausentes: {ausentes}")
    print("[INFO] Análise inicial concluída.")


def obter_indice_alvo(attributes, target_name=None):
    """
    Define qual coluna será o alvo da regressão.

    Regra:
        - Se o usuário informar --target, usa a coluna com esse nome.
        - Caso contrário, usa a última coluna do ARFF.

    No código da AV2 de regressão, o alvo era a última coluna, FPS.
    """
    if target_name:
        for i, attr in enumerate(attributes):
            if attr["name"].lower() == target_name.lower():
                return i
        raise ValueError(f"Coluna alvo não encontrada: {target_name}")
    return len(attributes) - 1


def separar_x_y_regressao(data, attributes, target_name=None, remover_strings=True):
    """
    Separa X e y para regressão.

    X = atributos preditores.
    y = variável alvo numérica.

    Por padrão, removemos atributos do tipo string, porque strings livres costumam ser
    identificadores ou textos com muitos valores únicos. Atributos nominais com conjunto
    fechado, como low/med/high/max, são mantidos e codificados numericamente.
    """
    print("[ETAPA] Separando variáveis preditoras (X) e alvo numérico (y)...")
    target_idx = obter_indice_alvo(attributes, target_name)
    target_attr = attributes[target_idx]

    X_raw = []
    y_raw = []
    feature_indices = []
    feature_attrs = []

    for idx, attr in enumerate(attributes):
        if idx == target_idx:
            continue
        if remover_strings and attr["type"] == "string":
            # TEORIA: strings livres podem ter muitos valores únicos e não representam
            # uma grandeza numérica útil. Por isso são removidas dos preditores.
            continue
        feature_indices.append(idx)
        feature_attrs.append(attr)

    linhas_ignoradas = 0
    for row in data:
        if len(row) <= target_idx:
            linhas_ignoradas += 1
            continue
        y_val = safe_float(row[target_idx])
        if is_missing(row[target_idx]) or np.isnan(y_val):
            # Como y é a resposta, linhas sem alvo não podem ser usadas no treino.
            linhas_ignoradas += 1
            continue

        X_raw.append([row[i] if i < len(row) else "?" for i in feature_indices])
        y_raw.append(y_val)

    print(f"[INFO] Variável alvo: {target_attr['name']} (coluna {target_idx})")
    print(f"[INFO] Atributos preditores usados antes da codificação: {len(feature_attrs)}")
    for i, attr in enumerate(feature_attrs, start=1):
        print(f"  {i:02d}. {attr['name']} ({attr['type']})")
    if linhas_ignoradas:
        print(f"[AVISO] Linhas ignoradas por ausência/problema no alvo: {linhas_ignoradas}")

    return X_raw, np.array(y_raw, dtype=float), feature_attrs, target_attr["name"]


# ============================================================
# ETAPA 3: PRÉ-PROCESSAMENTO MANUAL
# ============================================================

def ajustar_preprocessador_base(X_train_raw, feature_attrs):
    """
    Aprende como transformar cada coluna usando APENAS o treino.

    Numéricos:
        - converte para float;
        - valores ausentes são preenchidos com a média do treino.

    Categóricos/nominais:
        - faz Label Encoding manual;
        - categoria desconhecida no teste recebe -1.
    """
    preprocessador = []
    for j, attr in enumerate(feature_attrs):
        coluna = [row[j] for row in X_train_raw]
        if attr["type"] == "numeric":
            valores = np.array([safe_float(v) for v in coluna], dtype=float)
            media = np.nanmean(valores)
            if np.isnan(media):
                media = 0.0
            preprocessador.append({
                "name": attr["name"],
                "type": "numeric",
                "mean_fill": float(media),
            })
        else:
            categorias = sorted(set(str(v).strip().replace("'", "").replace('"', "") for v in coluna if not is_missing(v)))
            mapping = {cat: idx for idx, cat in enumerate(categorias)}
            preprocessador.append({
                "name": attr["name"],
                "type": "categorical",
                "mapping": mapping,
                "unknown_value": -1.0,
            })
    return preprocessador


def transformar_base(X_raw, preprocessador):
    """Aplica o preprocessador aprendido no treino."""
    X = np.zeros((len(X_raw), len(preprocessador)), dtype=float)
    for i, row in enumerate(X_raw):
        for j, config in enumerate(preprocessador):
            valor = row[j]
            if config["type"] == "numeric":
                v = safe_float(valor)
                X[i, j] = config["mean_fill"] if (is_missing(valor) or np.isnan(v)) else v
            else:
                valor_limpo = str(valor).strip().replace("'", "").replace('"', "")
                X[i, j] = config["unknown_value"] if is_missing(valor) else float(config["mapping"].get(valor_limpo, config["unknown_value"]))
    return X


def normalizar_treino_teste(X_train, X_test):
    """
    Normalização Z-score para X.

    Fórmula:
        X_norm = (X - média_treino) / desvio_treino

    A média e o desvio são calculados apenas no treino, evitando vazamento
    de dados do teste para o treinamento.
    """
    media = X_train.mean(axis=0)
    desvio = X_train.std(axis=0)
    desvio[desvio < EPS] = 1.0
    return (X_train - media) / desvio, (X_test - media) / desvio, media, desvio


def normalizar_y_treino_teste(y_train, y_test):
    """
    Normaliza o alvo y para facilitar o treino da MLP.

    A MLP treina melhor quando a saída também está em escala padronizada.
    Depois da predição, desfazemos a normalização para calcular métricas
    na escala original do problema.
    """
    media = float(np.mean(y_train))
    desvio = float(np.std(y_train))
    if desvio < EPS:
        desvio = 1.0
    y_train_norm = (y_train - media) / desvio
    y_test_norm = (y_test - media) / desvio
    return y_train_norm, y_test_norm, media, desvio


def desfazer_normalizacao_y(y_norm, media, desvio):
    """Volta as previsões da escala normalizada para a escala original."""
    return y_norm * desvio + media


def preprocessar_fold(X_train_raw, X_test_raw, feature_attrs):
    """Executa todo o pré-processamento de X dentro de um fold."""
    preprocessador = ajustar_preprocessador_base(X_train_raw, feature_attrs)
    X_train_base = transformar_base(X_train_raw, preprocessador)
    X_test_base = transformar_base(X_test_raw, preprocessador)
    X_train_norm, X_test_norm, _, _ = normalizar_treino_teste(X_train_base, X_test_base)
    feature_names = [attr["name"] for attr in feature_attrs]
    return X_train_norm, X_test_norm, feature_names


# ============================================================
# ETAPA 4: VALIDAÇÃO CRUZADA
# ============================================================

def dividir_k_folds(n_amostras, n_folds=5, seed=42):
    """Divide os índices em k folds manualmente."""
    indices = list(range(n_amostras))
    random.Random(seed).shuffle(indices)
    return [fold.tolist() for fold in np.array_split(np.array(indices, dtype=int), n_folds)]


# ============================================================
# ETAPA 5: FUNÇÕES DE ATIVAÇÃO
# ============================================================

def ativacao(z, nome):
    """
    Aplica função de ativação na camada oculta.

    ReLU:
        max(0, z). Costuma treinar bem e é simples.

    tanh:
        comprime valores entre -1 e 1. Pode funcionar bem com dados normalizados.

    sigmoid:
        comprime valores entre 0 e 1, mas pode saturar mais facilmente.
    """
    if nome == "relu":
        return np.maximum(0.0, z)
    if nome == "tanh":
        return np.tanh(z)
    if nome == "sigmoid":
        # Clip para evitar overflow em exp.
        z_clip = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z_clip))
    raise ValueError(f"Ativação desconhecida: {nome}")


def derivada_ativacao(a, nome):
    """
    Calcula a derivada da ativação usando a saída ativada 'a'.

    A derivada é necessária no backpropagation para ajustar os pesos.
    """
    if nome == "relu":
        return (a > 0).astype(float)
    if nome == "tanh":
        return 1.0 - a ** 2
    if nome == "sigmoid":
        return a * (1.0 - a)
    raise ValueError(f"Ativação desconhecida: {nome}")


# ============================================================
# ETAPA 6: MLP MANUAL PARA REGRESSÃO
# ============================================================

def inicializar_pesos(tamanhos_camadas, seed=42):
    """
    Inicializa pesos e bias da rede.

    tamanhos_camadas exemplo:
        [n_features, 16, 8, 1]

    Isso significa:
        entrada com n_features,
        primeira camada oculta com 16 neurônios,
        segunda camada oculta com 8 neurônios,
        saída com 1 neurônio.
    """
    rng = np.random.default_rng(seed)
    pesos = []
    bias = []

    for i in range(len(tamanhos_camadas) - 1):
        entrada = tamanhos_camadas[i]
        saida = tamanhos_camadas[i + 1]

        # Inicialização do tipo He/Xavier simplificada.
        # Ajuda a evitar valores iniciais grandes demais.
        escala = math.sqrt(2.0 / max(entrada, 1))
        W = rng.normal(0.0, escala, size=(entrada, saida))
        b = np.zeros((1, saida), dtype=float)
        pesos.append(W)
        bias.append(b)

    return pesos, bias


def forward_mlp(X, pesos, bias, ativacao_oculta):
    """
    Forward propagation.

    É o caminho da entrada até a saída.
    A rede recebe X, passa pelas camadas ocultas e gera uma previsão.

    Na regressão, a última camada é LINEAR, ou seja, não usamos ativação na saída.
    Isso permite prever qualquer valor numérico.
    """
    ativacoes = [X]
    z_values = []

    A = X
    for i in range(len(pesos)):
        Z = A @ pesos[i] + bias[i]
        z_values.append(Z)

        if i == len(pesos) - 1:
            # Saída linear para regressão.
            A = Z
        else:
            A = ativacao(Z, ativacao_oculta)

        ativacoes.append(A)

    return A, ativacoes, z_values


def backward_mlp(y_pred, y_true, pesos, ativacoes, ativacao_oculta):
    """
    Backpropagation manual para MSE.

    Objetivo:
        Calcular como cada peso contribuiu para o erro e gerar gradientes.

    Loss usada:
        MSE = média((y_pred - y_true)^2)
    """
    m = y_true.shape[0]
    y_true_col = y_true.reshape(-1, 1)

    # Derivada da MSE em relação à saída.
    dA = (2.0 / m) * (y_pred - y_true_col)

    grad_W = [None] * len(pesos)
    grad_b = [None] * len(pesos)

    dZ = dA  # saída linear: derivada = 1

    for i in reversed(range(len(pesos))):
        A_anterior = ativacoes[i]
        grad_W[i] = A_anterior.T @ dZ
        grad_b[i] = np.sum(dZ, axis=0, keepdims=True)

        if i > 0:
            dA_anterior = dZ @ pesos[i].T
            A_oculta = ativacoes[i]
            dZ = dA_anterior * derivada_ativacao(A_oculta, ativacao_oculta)

    return grad_W, grad_b


def aplicar_gradientes(pesos, bias, grad_W, grad_b, learning_rate):
    """Atualiza pesos e bias usando gradiente descendente."""
    for i in range(len(pesos)):
        # Clip simples para reduzir risco de explosão de gradientes.
        gW = np.clip(grad_W[i], -CLIP_GRADIENTE, CLIP_GRADIENTE)
        gb = np.clip(grad_b[i], -CLIP_GRADIENTE, CLIP_GRADIENTE)
        pesos[i] -= learning_rate * gW
        bias[i] -= learning_rate * gb


def treinar_mlp_regressao(
    X_train,
    y_train,
    hidden_layers,
    learning_rate=0.001,
    epochs=100,
    ativacao_oculta="relu",
    batch_size=64,
    seed=42,
):
    """
    Treina uma MLP manual para regressão.

    Hiperparâmetros principais:
        hidden_layers: topologia da rede, ex: [16] ou [16, 8]
        learning_rate: tamanho do passo de atualização dos pesos
        epochs: número de vezes que a rede percorre o treino
        ativacao_oculta: relu, tanh ou sigmoid
        batch_size: tamanho dos mini-lotes
    """
    n_features = X_train.shape[1]
    tamanhos_camadas = [n_features] + list(hidden_layers) + [1]
    pesos, bias = inicializar_pesos(tamanhos_camadas, seed=seed)

    rng = np.random.default_rng(seed)
    historico_loss = []

    for epoca in range(1, epochs + 1):
        indices = np.arange(X_train.shape[0])
        rng.shuffle(indices)

        perdas_epoca = []
        for inicio in range(0, len(indices), batch_size):
            idx = indices[inicio:inicio + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            y_pred, ativacoes, _ = forward_mlp(X_batch, pesos, bias, ativacao_oculta)
            loss = float(np.mean((y_pred.reshape(-1) - y_batch) ** 2))
            perdas_epoca.append(loss)

            grad_W, grad_b = backward_mlp(y_pred, y_batch, pesos, ativacoes, ativacao_oculta)
            aplicar_gradientes(pesos, bias, grad_W, grad_b, learning_rate)

        historico_loss.append(float(np.mean(perdas_epoca)) if perdas_epoca else 0.0)

    modelo = {
        "pesos": pesos,
        "bias": bias,
        "hidden_layers": hidden_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "ativacao": ativacao_oculta,
        "batch_size": batch_size,
        "historico_loss": historico_loss,
    }
    return modelo


def prever_mlp_regressao(modelo, X_test):
    """Gera previsões usando a MLP treinada."""
    y_pred, _, _ = forward_mlp(X_test, modelo["pesos"], modelo["bias"], modelo["ativacao"])
    return y_pred.reshape(-1)


# ============================================================
# ETAPA 7: MÉTRICAS DE REGRESSÃO MANUAIS
# ============================================================

def calcular_metricas_regressao(y_true, y_pred, n_features):
    """
    Calcula as métricas exigidas na AV3 para regressão.

    MSE:
        Média dos erros ao quadrado.

    RMSE:
        Raiz do MSE. Fica na mesma unidade do alvo.

    MAE:
        Média dos erros absolutos.

    R²:
        Quanto o modelo explica da variação do alvo.

    R² ajustado:
        R² penalizado pela quantidade de preditores.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot < EPS else float(1.0 - ss_res / ss_tot)

    n = len(y_true)
    p = n_features
    if n - p - 1 <= 0:
        r2_adj = 0.0
    else:
        r2_adj = float(1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1)))

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "r2_adj": r2_adj}


def media_desvio(valores):
    return float(np.mean(valores)), float(np.std(valores))


def formatar_media_desvio(media, desvio, casas=4):
    return f"{media:.{casas}f} ± {desvio:.{casas}f}"


# ============================================================
# ETAPA 8: EXECUÇÃO DOS EXPERIMENTOS
# ============================================================

def nome_configuracao(config):
    return (
        f"MLP hidden={config['hidden_layers']} "
        f"act={config['activation']} "
        f"lr={config['learning_rate']} "
        f"epochs={config['epochs']}"
    )


def montar_configuracoes(quick=False):
    """
    Define as arquiteturas avaliadas.

    A AV3 pede comparar diferentes:
        - camadas ocultas;
        - neurônios por camada;
        - taxa de aprendizado;
        - número de épocas;
        - funções de ativação.
    """
    if quick:
        return [
            {"hidden_layers": [8], "activation": "relu", "learning_rate": 0.001, "epochs": 10, "batch_size": 128},
            {"hidden_layers": [16, 8], "activation": "tanh", "learning_rate": 0.001, "epochs": 10, "batch_size": 128},
        ]

    return [
        # Uma camada oculta pequena.
        {"hidden_layers": [8], "activation": "relu", "learning_rate": 0.001, "epochs": 80, "batch_size": 128},

        # Uma camada oculta maior.
        {"hidden_layers": [16], "activation": "relu", "learning_rate": 0.001, "epochs": 80, "batch_size": 128},

        # Duas camadas ocultas, rede um pouco mais profunda.
        {"hidden_layers": [16, 8], "activation": "relu", "learning_rate": 0.001, "epochs": 80, "batch_size": 128},

        # Mesma topologia, mas mudando a função de ativação para tanh.
        {"hidden_layers": [16, 8], "activation": "tanh", "learning_rate": 0.001, "epochs": 80, "batch_size": 128},

        # Topologia maior e learning rate menor, para avaliar custo/desempenho.
        {"hidden_layers": [32, 16], "activation": "relu", "learning_rate": 0.0005, "epochs": 120, "batch_size": 128},
    ]


def executar_validacao_cruzada_mlp(X_raw, y, feature_attrs, configs, n_folds=5, seed=42):
    """Executa validação cruzada 5-Fold para cada configuração da MLP."""
    print(f"\n[ETAPA] Iniciando validação cruzada {n_folds}-Fold para MLP...")

    resultados = {}
    folds = dividir_k_folds(len(X_raw), n_folds=n_folds, seed=seed)
    todos_indices = set(range(len(X_raw)))

    for config_idx, config in enumerate(configs, start=1):
        nome = nome_configuracao(config)
        print("\n" + "=" * 100)
        print(f"CONFIGURAÇÃO {config_idx}/{len(configs)}: {nome}")
        print("=" * 100)

        resultados[nome] = {
            "mse": [], "rmse": [], "mae": [], "r2": [], "r2_adj": [],
            "train_time": [], "test_time": [], "final_loss": []
        }

        for fold_idx, test_indices in enumerate(folds, start=1):
            test_set = set(test_indices)
            train_indices = list(todos_indices - test_set)

            X_train_raw = [X_raw[i] for i in train_indices]
            X_test_raw = [X_raw[i] for i in test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            X_train, X_test, feature_names = preprocessar_fold(X_train_raw, X_test_raw, feature_attrs)
            y_train_norm, _, y_media, y_desvio = normalizar_y_treino_teste(y_train, y_test)

            t0 = time.perf_counter()
            modelo = treinar_mlp_regressao(
                X_train,
                y_train_norm,
                hidden_layers=config["hidden_layers"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                ativacao_oculta=config["activation"],
                batch_size=config["batch_size"],
                seed=seed + config_idx * 100 + fold_idx,
            )
            tempo_treino = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_norm = prever_mlp_regressao(modelo, X_test)
            y_pred = desfazer_normalizacao_y(pred_norm, y_media, y_desvio)
            tempo_teste = time.perf_counter() - t0

            metricas = calcular_metricas_regressao(y_test, y_pred, n_features=X_train.shape[1])

            for chave in ["mse", "rmse", "mae", "r2", "r2_adj"]:
                resultados[nome][chave].append(metricas[chave])
            resultados[nome]["train_time"].append(tempo_treino)
            resultados[nome]["test_time"].append(tempo_teste)
            resultados[nome]["final_loss"].append(modelo["historico_loss"][-1] if modelo["historico_loss"] else 0.0)

            print(
                f"  Fold {fold_idx}/{n_folds}... "
                f"MSE={metricas['mse']:.4f} RMSE={metricas['rmse']:.4f} "
                f"MAE={metricas['mae']:.4f} R2={metricas['r2']:.4f} "
                f"treino={tempo_treino:.3f}s teste={tempo_teste:.3f}s "
                f"loss_final={resultados[nome]['final_loss'][-1]:.4f}"
            )

    return resultados


def resumir_resultados(resultados):
    """Calcula média e desvio padrão de cada métrica."""
    resumo = {}
    for nome, metricas in resultados.items():
        resumo[nome] = {}
        for chave, valores in metricas.items():
            media, desvio = media_desvio(valores)
            resumo[nome][chave] = {"mean": media, "std": desvio}
    return resumo


def ordenar_por_melhor_modelo(resumo):
    """
    Ordena principalmente por menor RMSE e, em empate, maior R².

    Para regressão:
        - RMSE menor é melhor;
        - MAE menor é melhor;
        - MSE menor é melhor;
        - R² maior é melhor.
    """
    return sorted(
        resumo.keys(),
        key=lambda nome: (resumo[nome]["rmse"]["mean"], -resumo[nome]["r2"]["mean"], resumo[nome]["train_time"]["mean"])
    )


def imprimir_tabela_resultados(resumo):
    print("\n" + "=" * 150)
    print("TABELA COMPARATIVA - MLP PARA REGRESSÃO")
    print("=" * 150)
    print(
        f"{'Configuração':<58} "
        f"{'MSE':>16} {'RMSE':>16} {'MAE':>16} {'R2':>16} {'R2 Ajust.':>16} "
        f"{'T.Treino(s)':>16} {'T.Teste(s)':>16}"
    )
    print("─" * 150)

    for nome in ordenar_por_melhor_modelo(resumo):
        r = resumo[nome]
        print(
            f"{nome:<58} "
            f"{formatar_media_desvio(r['mse']['mean'], r['mse']['std']):>16} "
            f"{formatar_media_desvio(r['rmse']['mean'], r['rmse']['std']):>16} "
            f"{formatar_media_desvio(r['mae']['mean'], r['mae']['std']):>16} "
            f"{formatar_media_desvio(r['r2']['mean'], r['r2']['std']):>16} "
            f"{formatar_media_desvio(r['r2_adj']['mean'], r['r2_adj']['std']):>16} "
            f"{formatar_media_desvio(r['train_time']['mean'], r['train_time']['std'], casas=3):>16} "
            f"{formatar_media_desvio(r['test_time']['mean'], r['test_time']['std'], casas=4):>16}"
        )
    print("=" * 150)


def gerar_relatorio_texto(resumo, target_name, n_instancias, n_features, configs):
    """Gera o texto que será salvo em resultados_mlp.txt."""
    linhas = []
    linhas.append("AV3 - REGRESSÃO COM MLP MANUAL")
    linhas.append("=" * 100)
    linhas.append(f"Alvo da regressão: {target_name}")
    linhas.append(f"Instâncias usadas: {n_instancias}")
    linhas.append(f"Preditores usados: {n_features}")
    linhas.append(f"Configurações avaliadas: {len(configs)}")
    linhas.append("")
    linhas.append("Tabela comparativa:")
    linhas.append("-" * 100)

    cab = (
        f"{'Configuração':<58} {'MSE':>16} {'RMSE':>16} {'MAE':>16} "
        f"{'R2':>16} {'R2 Ajust.':>16} {'T.Treino(s)':>16} {'T.Teste(s)':>16}"
    )
    linhas.append(cab)
    linhas.append("-" * 150)

    for nome in ordenar_por_melhor_modelo(resumo):
        r = resumo[nome]
        linhas.append(
            f"{nome:<58} "
            f"{formatar_media_desvio(r['mse']['mean'], r['mse']['std']):>16} "
            f"{formatar_media_desvio(r['rmse']['mean'], r['rmse']['std']):>16} "
            f"{formatar_media_desvio(r['mae']['mean'], r['mae']['std']):>16} "
            f"{formatar_media_desvio(r['r2']['mean'], r['r2']['std']):>16} "
            f"{formatar_media_desvio(r['r2_adj']['mean'], r['r2_adj']['std']):>16} "
            f"{formatar_media_desvio(r['train_time']['mean'], r['train_time']['std'], casas=3):>16} "
            f"{formatar_media_desvio(r['test_time']['mean'], r['test_time']['std'], casas=4):>16}"
        )

    melhor = ordenar_por_melhor_modelo(resumo)[0]
    r = resumo[melhor]

    linhas.append("")
    linhas.append("MELHOR CONFIGURAÇÃO")
    linhas.append("=" * 100)
    linhas.append(f"Melhor configuração pelo menor RMSE médio: {melhor}")
    linhas.append(f"RMSE médio: {r['rmse']['mean']:.4f} ± {r['rmse']['std']:.4f}")
    linhas.append(f"MAE médio: {r['mae']['mean']:.4f} ± {r['mae']['std']:.4f}")
    linhas.append(f"R2 médio: {r['r2']['mean']:.4f} ± {r['r2']['std']:.4f}")
    linhas.append(f"R2 ajustado médio: {r['r2_adj']['mean']:.4f} ± {r['r2_adj']['std']:.4f}")
    linhas.append(f"Tempo médio de treino: {r['train_time']['mean']:.3f}s")
    linhas.append(f"Tempo médio de teste: {r['test_time']['mean']:.4f}s")

    linhas.append("")
    linhas.append("INTERPRETAÇÃO PARA OS SLIDES")
    linhas.append("=" * 100)
    linhas.append("1. Foram avaliadas diferentes topologias de MLP, variando camadas ocultas, neurônios, ativação, learning rate e épocas.")
    linhas.append("2. A saída da rede é linear, pois o problema é de regressão e o objetivo é prever um valor numérico contínuo.")
    linhas.append("3. O RMSE e o MAE indicam o tamanho médio do erro de previsão na escala original do alvo.")
    linhas.append("4. O R2 mostra quanto da variação do alvo é explicada pelo modelo; valores maiores indicam melhor ajuste.")
    linhas.append("5. O R2 ajustado penaliza modelos com muitos preditores, ajudando a avaliar generalização.")
    linhas.append("6. Arquiteturas maiores podem reduzir erro, mas normalmente aumentam custo computacional.")
    linhas.append("7. A melhor escolha final deve equilibrar erro baixo, bom R2/R2 ajustado e tempo de execução aceitável.")

    return "\n".join(linhas)


def salvar_relatorio(texto, caminho=RESULTADOS_ARQUIVO):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"\n[INFO] Relatório salvo em: {caminho}")


# ============================================================
# ETAPA 9: MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AV3 - Regressão com MLP manual")
    parser.add_argument("--arquivo", default=ARQUIVO_PADRAO, help="Caminho para o arquivo ARFF de regressão")
    parser.add_argument("--target", default=None, help="Nome da coluna alvo. Se omitido, usa a última coluna.")
    parser.add_argument("--quick", action="store_true", help="Modo rápido para testar se o pipeline funciona")
    parser.add_argument("--folds", type=int, default=None, help="Número de folds. Padrão: 5; quick: 2")
    args = parser.parse_args()

    inicio_total = time.perf_counter()
    n_folds = args.folds if args.folds is not None else (2 if args.quick else N_FOLDS_PADRAO)
    configs = montar_configuracoes(quick=args.quick)

    print("=" * 100)
    print("AV3 - REGRESSÃO COM MULTI LAYER PERCEPTRON (MLP) MANUAL")
    print("=" * 100)
    print(f"Dataset: {args.arquivo}")
    print(f"Target: {args.target if args.target else 'última coluna do ARFF'}")
    print(f"Folds: {n_folds}")
    print(f"Configurações MLP: {len(configs)}")
    print("=" * 100)

    relation, attributes, data = carregar_arff(args.arquivo)
    analisar_dataset(relation, attributes, data)

    X_raw, y, feature_attrs, target_name = separar_x_y_regressao(data, attributes, target_name=args.target)

    if len(X_raw) == 0:
        raise ValueError("Nenhuma linha válida foi carregada. Verifique o dataset e a coluna alvo.")

    print("\n[INFO] Resumo após preparação inicial:")
    print(f"  Instâncias válidas: {len(X_raw)}")
    print(f"  Preditores antes da codificação: {len(feature_attrs)}")
    print(f"  Alvo: {target_name}")
    print(f"  Média do alvo: {np.mean(y):.4f}")
    print(f"  Desvio padrão do alvo: {np.std(y):.4f}")

    resultados = executar_validacao_cruzada_mlp(
        X_raw=X_raw,
        y=y,
        feature_attrs=feature_attrs,
        configs=configs,
        n_folds=n_folds,
        seed=SEED,
    )

    resumo = resumir_resultados(resultados)
    imprimir_tabela_resultados(resumo)

    # Para contar features reais, preprocessa uma pequena divisão simples.
    folds_tmp = dividir_k_folds(len(X_raw), n_folds=n_folds, seed=SEED)
    test_idx_tmp = set(folds_tmp[0])
    train_idx_tmp = list(set(range(len(X_raw))) - test_idx_tmp)
    X_train_tmp = [X_raw[i] for i in train_idx_tmp]
    X_test_tmp = [X_raw[i] for i in folds_tmp[0]]
    X_train_norm_tmp, _, feature_names = preprocessar_fold(X_train_tmp, X_test_tmp, feature_attrs)

    relatorio = gerar_relatorio_texto(
        resumo=resumo,
        target_name=target_name,
        n_instancias=len(X_raw),
        n_features=X_train_norm_tmp.shape[1],
        configs=configs,
    )
    salvar_relatorio(relatorio)

    print("\n" + "=" * 100)
    print("INTERPRETAÇÃO RÁPIDA")
    print("=" * 100)
    melhor = ordenar_por_melhor_modelo(resumo)[0]
    print(f"Melhor configuração pelo menor RMSE: {melhor}")
    print("Para os slides, compare RMSE/MAE/R2 com tempo de treino e teste.")
    print("Se uma rede maior melhorar pouco o erro mas custar muito mais tempo, ela pode não ser o melhor equilíbrio.")
    print(f"\n[INFO] Tempo total de execução: {time.perf_counter() - inicio_total:.1f}s")
    print("Pipeline da regressão AV3 concluído com sucesso!")


if __name__ == "__main__":
    main()

"""
AV3 - CLASSIFICAÇÃO COM PERCEPTRON SIMPLES
Disciplina: Inteligência Artificial Computacional

Este arquivo é a continuação da parte de classificação feita na AV2.
Ele reaproveita a leitura do ARFF, o pré-processamento, as features derivadas,
a normalização e a validação cruzada do arquivo:

    classificacao_videogames_cenario_c.py

Novo modelo implementado na AV3:

    Perceptron Simples Multiclasse

Por que multiclasse?
    Na AV2/AV3, a variável alvo é Genre. Essa variável possui várias classes
    possíveis, como Action, Sports, Racing, Shooter, Puzzle etc. O Perceptron
    clássico é naturalmente binário, mas aqui usamos uma adaptação multiclasse:

        - cada classe possui um vetor de pesos;
        - para uma amostra, calculamos uma pontuação para cada classe;
        - a classe com maior pontuação é a previsão;
        - se a previsão estiver errada, atualizamos os pesos da classe correta
          e da classe prevista incorretamente.

Regras respeitadas:
    - Não usa pandas.
    - Não usa scikit-learn.
    - Não usa modelos prontos de aprendizado de máquina.
    - Perceptron implementado manualmente.
    - Validação cruzada k-fold manual, reaproveitada da AV2.
    - Métricas implementadas manualmente.

Métricas exigidas na AV3 para classificação:
    - Matriz de confusão;
    - Acurácia;
    - Sensibilidade / Recall;
    - Especificidade;
    - Precisão;
    - F1-Score.

Como executar:
    Coloque este arquivo na mesma pasta de:
        - videogamesales.arff
        - classificacao_videogames_cenario_c.py

    Execução normal:
        python classificacao_perceptron_av3.py

    Teste rápido:
        python classificacao_perceptron_av3.py --quick

    Exemplo escolhendo hiperparâmetros manualmente:
        python classificacao_perceptron_av3.py --learning-rates 0.001,0.01,0.1 --epochs 10,30,50
"""

import argparse
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# IMPORTAÇÃO DA BASE DA AV2
# ============================================================
# A AV3 pede reaproveitar o que já foi construído na AV2. Por isso,
# importamos as funções já prontas da classificação do Cenário C.
# Esse arquivo precisa estar na mesma pasta deste script.

try:
    from classificacao_videogames_cenario_c import (
        ARQUIVO_PADRAO,
        SEED,
        TARGET_NAME,
        analisar_dataset,
        carregar_arff,
        codificar_y,
        dividir_k_folds,
        matriz_confusao,
        mostrar_distribuicao_classes,
        preprocessar_fold_cenario_c,
        separar_x_y_cenario_c,
    )
except ImportError as erro:
    raise ImportError(
        "Não consegui importar 'classificacao_videogames_cenario_c.py'.\n"
        "Verifique se este arquivo está na mesma pasta de classificacao_perceptron_av3.py.\n"
        f"Erro original: {erro}"
    )


# ============================================================
# CONFIGURAÇÕES DA AV3
# ============================================================

N_FOLDS_PADRAO = 5

# Hiperparâmetros padrão para investigar o impacto no Perceptron.
# A AV3 pede testar diferentes taxas de aprendizado e diferentes épocas.
LEARNING_RATES_PADRAO = [0.001, 0.01, 0.1]
EPOCHS_PADRAO = [10, 30, 50]

SAIDA_PADRAO = os.path.join("resultados_av3", "resultados_perceptron.txt")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def parse_lista_float(texto: str) -> List[float]:
    """Converte texto como '0.001,0.01,0.1' em lista de floats."""
    return [float(x.strip()) for x in texto.split(",") if x.strip()]


def parse_lista_int(texto: str) -> List[int]:
    """Converte texto como '10,30,50' em lista de inteiros."""
    return [int(x.strip()) for x in texto.split(",") if x.strip()]


def media_desvio(valores: List[float]) -> Tuple[float, float]:
    """Calcula média e desvio padrão de uma lista de valores."""
    arr = np.array(valores, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def formatar_media_desvio(media: float, desvio: float, casas: int = 4) -> str:
    """Formata valores no padrão média ± desvio."""
    return f"{media:.{casas}f} ± {desvio:.{casas}f}"


# ============================================================
# PERCEPTRON SIMPLES MULTICLASSE
# ============================================================

def treinar_perceptron_multiclasse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    learning_rate: float,
    epochs: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Treina um Perceptron Simples Multiclasse manualmente.

    Ideia do modelo:
        Cada classe possui um vetor de pesos. Para cada amostra x, calculamos:

            score_classe = w_classe · x + b_classe

        A classe escolhida é a que tem maior score.

    Atualização quando erra:
        Se a classe correta é y_real e a classe prevista é y_pred:

            W[y_real] += learning_rate * x
            b[y_real] += learning_rate

            W[y_pred] -= learning_rate * x
            b[y_pred] -= learning_rate

        Ou seja:
            - aumentamos a pontuação futura da classe correta;
            - reduzimos a pontuação futura da classe errada.

    Hiperparâmetros:
        learning_rate:
            Define o tamanho do ajuste dos pesos a cada erro.
            Taxa muito baixa pode aprender devagar.
            Taxa muito alta pode causar instabilidade.

        epochs:
            Define quantas vezes o algoritmo percorre todo o conjunto de treino.
            Poucas épocas podem gerar pouco aprendizado.
            Muitas épocas podem aumentar custo computacional e, em alguns casos,
            piorar generalização.
    """
    rng = random.Random(seed)

    n_amostras, n_features = X_train.shape

    # W guarda os pesos de cada classe.
    # Formato: n_classes x n_features.
    W = np.zeros((n_classes, n_features), dtype=float)

    # b é o bias de cada classe.
    b = np.zeros(n_classes, dtype=float)

    historico_erros = []

    indices = list(range(n_amostras))

    for epoch in range(epochs):
        rng.shuffle(indices)
        erros = 0

        for idx in indices:
            x = X_train[idx]
            y_real = int(y_train[idx])

            # Calcula uma pontuação para cada classe.
            scores = W @ x + b

            # Predição: classe com maior pontuação.
            y_pred = int(np.argmax(scores))

            if y_pred != y_real:
                erros += 1

                # Reforça a classe correta.
                W[y_real] += learning_rate * x
                b[y_real] += learning_rate

                # Penaliza a classe errada.
                W[y_pred] -= learning_rate * x
                b[y_pred] -= learning_rate

        historico_erros.append(erros)

        # Parada antecipada simples: se não houve erro, o modelo separou o treino
        # nessa época. Em problemas reais e multiclasse isso pode não acontecer.
        if erros == 0:
            break

    return {
        "W": W,
        "b": b,
        "historico_erros": np.array(historico_erros, dtype=int),
        "learning_rate": learning_rate,
        "epochs_solicitadas": epochs,
        "epochs_executadas": len(historico_erros),
    }


def prever_perceptron_multiclasse(modelo: Dict[str, np.ndarray], X_test: np.ndarray) -> np.ndarray:
    """
    Realiza predições com o Perceptron Multiclasse.

    Para cada amostra, calcula a pontuação de todas as classes:

        scores = X @ W.T + b

    Depois escolhe a classe com maior pontuação.
    """
    W = modelo["W"]
    b = modelo["b"]

    scores = X_test @ W.T + b
    predicoes = np.argmax(scores, axis=1)

    return predicoes.astype(int)


# ============================================================
# MÉTRICAS MANUAIS DA CLASSIFICAÇÃO
# ============================================================

def calcular_metricas_av3(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    """
    Calcula as métricas exigidas na AV3 manualmente.

    Métricas:
        - accuracy: acertos / total;
        - precision macro: média da precisão de todas as classes;
        - recall macro: média do recall/sensibilidade de todas as classes;
        - specificity macro: média da especificidade de todas as classes;
        - f1 macro: média do F1 de todas as classes.

    Por que macro average?
        O problema tem várias classes de gênero e elas são desbalanceadas.
        Macro average calcula a métrica classe por classe e depois tira a média,
        dando peso igual para cada gênero.
    """
    cm = matriz_confusao(y_true, y_pred, n_classes)
    total = int(np.sum(cm))

    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    precisions = []
    recalls = []
    specificities = []
    f1s = []

    for classe in range(n_classes):
        tp = cm[classe, classe]
        fp = np.sum(cm[:, classe]) - tp
        fn = np.sum(cm[classe, :]) - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        specificities.append(float(specificity))
        f1s.append(float(f1))

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "specificity": float(np.mean(specificities)),
        "f1": float(np.mean(f1s)),
        "confusion_matrix": cm,
    }


# ============================================================
# VALIDAÇÃO CRUZADA E EXPERIMENTOS
# ============================================================

def preparar_folds(
    X_raw: List[List[str]],
    y: np.ndarray,
    feature_attrs: List[dict],
    n_folds: int,
    seed: int,
):
    """
    Prepara os folds da validação cruzada.

    Importante:
        O pré-processamento é feito dentro de cada fold.
        Isso evita vazamento de dados, porque as estatísticas do teste não são usadas
        para transformar o treino.
    """
    folds = dividir_k_folds(len(y), n_folds=n_folds, seed=seed)
    todos_indices = set(range(len(y)))
    folds_preparados = []
    feature_names_final = None

    for fold_idx, test_indices in enumerate(folds, start=1):
        test_set = set(test_indices)
        train_indices = sorted(list(todos_indices - test_set))
        test_indices = sorted(test_indices)

        X_train_raw = [X_raw[i] for i in train_indices]
        X_test_raw = [X_raw[i] for i in test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        X_train, X_test, feature_names_final = preprocessar_fold_cenario_c(
            X_train_raw,
            X_test_raw,
            feature_attrs,
        )

        folds_preparados.append({
            "fold": fold_idx,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })

    return folds_preparados, feature_names_final


def adicionar_resultado(resultados: Dict[str, list], config_nome: str, metricas: Dict[str, float], tempo_treino: float, tempo_teste: float):
    """Guarda resultado de um fold para uma determinada configuração."""
    if config_nome not in resultados:
        resultados[config_nome] = []

    resultados[config_nome].append({
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "specificity": metricas["specificity"],
        "f1": metricas["f1"],
        "train_time": tempo_treino,
        "test_time": tempo_teste,
        "confusion_matrix": metricas["confusion_matrix"],
    })


def executar_experimentos_perceptron(
    X_raw: List[List[str]],
    y: np.ndarray,
    feature_attrs: List[dict],
    n_classes: int,
    learning_rates: List[float],
    epochs_list: List[int],
    n_folds: int,
    seed: int,
):
    """
    Executa validação cruzada para várias configurações do Perceptron.

    A AV3 pede investigar o impacto de:
        - taxa de aprendizado;
        - número de épocas.

    Por isso, testamos todas as combinações entre learning_rates e epochs_list.
    """
    print("\n[INFO] Preparando folds e pré-processamento...")
    folds_preparados, feature_names_final = preparar_folds(
        X_raw=X_raw,
        y=y,
        feature_attrs=feature_attrs,
        n_folds=n_folds,
        seed=seed,
    )

    print(f"[INFO] Total de preditores após features derivadas: {len(feature_names_final)}")
    print("[INFO] Features finais:")
    for i, nome in enumerate(feature_names_final, start=1):
        print(f"  {i:02d}. {nome}")

    resultados = {}

    for lr in learning_rates:
        for epochs in epochs_list:
            config_nome = f"Perceptron lr={lr} epochs={epochs}"
            print("\n" + "=" * 90)
            print(f"CONFIGURAÇÃO: {config_nome}")
            print("=" * 90)

            for fold_data in folds_preparados:
                fold = fold_data["fold"]
                X_train = fold_data["X_train"]
                X_test = fold_data["X_test"]
                y_train = fold_data["y_train"]
                y_test = fold_data["y_test"]

                print(f"  Fold {fold}/{n_folds}...", end=" ")

                inicio_treino = time.perf_counter()
                modelo = treinar_perceptron_multiclasse(
                    X_train=X_train,
                    y_train=y_train,
                    n_classes=n_classes,
                    learning_rate=lr,
                    epochs=epochs,
                    seed=seed + fold,
                )
                tempo_treino = time.perf_counter() - inicio_treino

                inicio_teste = time.perf_counter()
                y_pred = prever_perceptron_multiclasse(modelo, X_test)
                tempo_teste = time.perf_counter() - inicio_teste

                metricas = calcular_metricas_av3(y_test, y_pred, n_classes)
                adicionar_resultado(resultados, config_nome, metricas, tempo_treino, tempo_teste)

                erros_ultima_epoca = int(modelo["historico_erros"][-1]) if len(modelo["historico_erros"]) else -1
                print(
                    f"acc={metricas['accuracy']:.4f} "
                    f"f1={metricas['f1']:.4f} "
                    f"treino={tempo_treino:.3f}s "
                    f"teste={tempo_teste:.3f}s "
                    f"erros_ult_epoca={erros_ultima_epoca}"
                )

    return resultados, feature_names_final


# ============================================================
# RESUMO, TABELAS E RELATÓRIO
# ============================================================

def resumir_resultados(resultados: Dict[str, list]) -> Dict[str, dict]:
    """Calcula média e desvio padrão das métricas em todos os folds."""
    resumo = {}

    for config_nome, linhas in resultados.items():
        resumo[config_nome] = {}

        for chave in ["accuracy", "precision", "recall", "specificity", "f1", "train_time", "test_time"]:
            valores = [linha[chave] for linha in linhas]
            media, desvio = media_desvio(valores)
            resumo[config_nome][chave] = {"mean": media, "std": desvio}

        # Soma das matrizes de confusão dos folds.
        # Isso gera uma matriz agregada da configuração.
        matriz_total = None
        for linha in linhas:
            cm = linha["confusion_matrix"]
            matriz_total = cm.copy() if matriz_total is None else matriz_total + cm

        resumo[config_nome]["confusion_matrix"] = matriz_total

    return resumo


def obter_melhor_configuracao(resumo: Dict[str, dict]) -> str:
    """
    Seleciona a melhor configuração pelo maior F1-score médio.

    Por que F1?
        O dataset possui várias classes e distribuição desbalanceada.
        O F1 macro equilibra precisão e recall, sendo mais informativo que olhar
        apenas a acurácia.
    """
    return max(resumo.keys(), key=lambda nome: resumo[nome]["f1"]["mean"])


def imprimir_tabela_resumo(resumo: Dict[str, dict]):
    """Imprime tabela comparativa das configurações do Perceptron."""
    print("\n" + "=" * 130)
    print("TABELA COMPARATIVA - PERCEPTRON SIMPLES MULTICLASSE")
    print("=" * 130)
    print(
        f"{'Configuração':<38} "
        f"{'Acurácia':>17} "
        f"{'Precisão':>17} "
        f"{'Recall/Sens.':>17} "
        f"{'Especific.':>17} "
        f"{'F1-Score':>17} "
        f"{'T.Treino(s)':>17} "
        f"{'T.Teste(s)':>17}"
    )
    print("─" * 130)

    for nome, dados in sorted(resumo.items(), key=lambda item: item[1]["f1"]["mean"], reverse=True):
        print(
            f"{nome:<38} "
            f"{formatar_media_desvio(dados['accuracy']['mean'], dados['accuracy']['std']):>17} "
            f"{formatar_media_desvio(dados['precision']['mean'], dados['precision']['std']):>17} "
            f"{formatar_media_desvio(dados['recall']['mean'], dados['recall']['std']):>17} "
            f"{formatar_media_desvio(dados['specificity']['mean'], dados['specificity']['std']):>17} "
            f"{formatar_media_desvio(dados['f1']['mean'], dados['f1']['std']):>17} "
            f"{formatar_media_desvio(dados['train_time']['mean'], dados['train_time']['std'], casas=3):>17} "
            f"{formatar_media_desvio(dados['test_time']['mean'], dados['test_time']['std'], casas=3):>17}"
        )

    print("=" * 130)


def matriz_confusao_para_texto(cm: np.ndarray, int_to_class: Dict[int, str]) -> str:
    """Transforma matriz de confusão em texto legível para salvar no arquivo."""
    linhas = []
    classes = [int_to_class[i] for i in range(len(int_to_class))]

    linhas.append("Matriz de confusão agregada da melhor configuração")
    linhas.append("Linhas = classe real | Colunas = classe prevista")
    linhas.append("")

    cabecalho = "Classe real \\ prevista".ljust(24)
    for classe in classes:
        cabecalho += classe[:10].rjust(12)
    linhas.append(cabecalho)
    linhas.append("-" * len(cabecalho))

    for i, classe in enumerate(classes):
        linha = classe[:22].ljust(24)
        for valor in cm[i]:
            linha += str(int(valor)).rjust(12)
        linhas.append(linha)

    return "\n".join(linhas)


def salvar_relatorio(
    caminho_saida: str,
    resumo: Dict[str, dict],
    melhor_config: str,
    int_to_class: Dict[int, str],
    feature_names_final: List[str],
    learning_rates: List[float],
    epochs_list: List[int],
    n_folds: int,
):
    """Salva um relatório .txt com tabela, matriz de confusão e interpretação."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("AV3 - CLASSIFICAÇÃO COM PERCEPTRON SIMPLES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {ARQUIVO_PADRAO}\n")
        f.write(f"Variável alvo: {TARGET_NAME}\n")
        f.write(f"Validação cruzada: {n_folds}-Fold\n")
        f.write(f"Learning rates testados: {learning_rates}\n")
        f.write(f"Épocas testadas: {epochs_list}\n")
        f.write("\n")

        f.write("PREDITORES UTILIZADOS\n")
        f.write("-" * 80 + "\n")
        f.write("A base usa o Cenário C da AV2: remove Name e Global_Sales,\n")
        f.write("mantém Genre como alvo e cria features derivadas a partir de vendas regionais, Rank e Year.\n\n")
        for i, nome in enumerate(feature_names_final, start=1):
            f.write(f"{i:02d}. {nome}\n")
        f.write("\n")

        f.write("TABELA COMPARATIVA\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Configuração':<38} "
            f"{'Acurácia':>17} "
            f"{'Precisão':>17} "
            f"{'Recall':>17} "
            f"{'Especific.':>17} "
            f"{'F1':>17} "
            f"{'Treino(s)':>17} "
            f"{'Teste(s)':>17}\n"
        )
        f.write("-" * 160 + "\n")

        for nome, dados in sorted(resumo.items(), key=lambda item: item[1]["f1"]["mean"], reverse=True):
            f.write(
                f"{nome:<38} "
                f"{formatar_media_desvio(dados['accuracy']['mean'], dados['accuracy']['std']):>17} "
                f"{formatar_media_desvio(dados['precision']['mean'], dados['precision']['std']):>17} "
                f"{formatar_media_desvio(dados['recall']['mean'], dados['recall']['std']):>17} "
                f"{formatar_media_desvio(dados['specificity']['mean'], dados['specificity']['std']):>17} "
                f"{formatar_media_desvio(dados['f1']['mean'], dados['f1']['std']):>17} "
                f"{formatar_media_desvio(dados['train_time']['mean'], dados['train_time']['std'], casas=3):>17} "
                f"{formatar_media_desvio(dados['test_time']['mean'], dados['test_time']['std'], casas=3):>17}\n"
            )

        f.write("\n")
        f.write(f"Melhor configuração pelo F1-score médio: {melhor_config}\n")
        f.write(
            f"F1 médio: {resumo[melhor_config]['f1']['mean']:.4f} ± "
            f"{resumo[melhor_config]['f1']['std']:.4f}\n"
        )
        f.write(
            f"Acurácia média: {resumo[melhor_config]['accuracy']['mean']:.4f} ± "
            f"{resumo[melhor_config]['accuracy']['std']:.4f}\n\n"
        )

        f.write(matriz_confusao_para_texto(resumo[melhor_config]["confusion_matrix"], int_to_class))
        f.write("\n\n")

        f.write("INTERPRETAÇÃO PARA OS SLIDES\n")
        f.write("-" * 80 + "\n")
        f.write("1. O Perceptron foi avaliado com diferentes taxas de aprendizado e épocas.\n")
        f.write("2. A taxa de aprendizado controla o tamanho do ajuste dos pesos a cada erro.\n")
        f.write("3. O número de épocas controla quantas vezes o algoritmo percorre todo o treino.\n")
        f.write("4. A melhor configuração foi escolhida pelo F1-score macro, pois o problema é multiclasse e desbalanceado.\n")
        f.write("5. A especificidade foi calculada em estratégia one-vs-rest para cada classe e depois feita a média macro.\n")
        f.write("6. O tempo de treino cresce quando aumentamos o número de épocas.\n")
        f.write("7. O tempo de teste tende a ser baixo, pois a predição é apenas uma multiplicação matricial.\n")

    print(f"\n[INFO] Relatório salvo em: {caminho_saida}")


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AV3 - Classificação com Perceptron Simples Multiclasse")
    parser.add_argument("--arquivo", default=ARQUIVO_PADRAO, help="Caminho do arquivo ARFF de classificação")
    parser.add_argument("--learning-rates", default=",".join(map(str, LEARNING_RATES_PADRAO)), help="Lista de learning rates, ex: 0.001,0.01,0.1")
    parser.add_argument("--epochs", default=",".join(map(str, EPOCHS_PADRAO)), help="Lista de épocas, ex: 10,30,50")
    parser.add_argument("--folds", type=int, default=N_FOLDS_PADRAO, help="Número de folds da validação cruzada")
    parser.add_argument("--saida", default=SAIDA_PADRAO, help="Arquivo .txt onde os resultados serão salvos")
    parser.add_argument("--quick", action="store_true", help="Modo rápido para testar se o código funciona")
    parser.add_argument("--max-amostras", type=int, default=None, help="Limita o número de amostras para testes rápidos")
    parser.add_argument("--seed", type=int, default=SEED, help="Semente aleatória")

    args = parser.parse_args()

    learning_rates = parse_lista_float(args.learning_rates)
    epochs_list = parse_lista_int(args.epochs)
    n_folds = args.folds

    if args.quick:
        print("[MODO QUICK] Rodando teste reduzido para validar o pipeline.")
        learning_rates = [0.01, 0.1]
        epochs_list = [5, 10]
        n_folds = 3
        if args.max_amostras is None:
            args.max_amostras = 3000

    print("=" * 90)
    print("AV3 - CLASSIFICAÇÃO DE GÊNERO DE VIDEOGAMES COM PERCEPTRON SIMPLES")
    print("=" * 90)
    print(f"Dataset: {args.arquivo}")
    print(f"Alvo: {TARGET_NAME}")
    print(f"Learning rates: {learning_rates}")
    print(f"Épocas: {epochs_list}")
    print(f"Folds: {n_folds}")
    print("=" * 90)

    inicio_total = time.perf_counter()

    relation, attributes, data = carregar_arff(args.arquivo)
    analisar_dataset(relation, attributes, data)

    X_raw, y_raw, feature_attrs = separar_x_y_cenario_c(data, attributes)

    if args.max_amostras is not None and args.max_amostras < len(y_raw):
        print(f"[INFO] Limitando dataset para {args.max_amostras} amostras por --max-amostras.")
        rng = random.Random(args.seed)
        indices = list(range(len(y_raw)))
        rng.shuffle(indices)
        indices = sorted(indices[:args.max_amostras])
        X_raw = [X_raw[i] for i in indices]
        y_raw = [y_raw[i] for i in indices]

    y, class_to_int, int_to_class = codificar_y(y_raw)
    n_classes = len(class_to_int)

    print(f"\n[INFO] Quantidade de classes: {n_classes}")
    print(f"[INFO] Classes: {list(class_to_int.keys())}\n")
    mostrar_distribuicao_classes(y_raw)

    resultados, feature_names_final = executar_experimentos_perceptron(
        X_raw=X_raw,
        y=y,
        feature_attrs=feature_attrs,
        n_classes=n_classes,
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        n_folds=n_folds,
        seed=args.seed,
    )

    resumo = resumir_resultados(resultados)
    melhor_config = obter_melhor_configuracao(resumo)

    imprimir_tabela_resumo(resumo)

    print("\n" + "=" * 90)
    print("MELHOR CONFIGURAÇÃO")
    print("=" * 90)
    print(f"Melhor configuração pelo F1-score macro: {melhor_config}")
    print(f"F1-score médio: {resumo[melhor_config]['f1']['mean']:.4f} ± {resumo[melhor_config]['f1']['std']:.4f}")
    print(f"Acurácia média: {resumo[melhor_config]['accuracy']['mean']:.4f} ± {resumo[melhor_config]['accuracy']['std']:.4f}")
    print(f"Recall/Sensibilidade média: {resumo[melhor_config]['recall']['mean']:.4f} ± {resumo[melhor_config]['recall']['std']:.4f}")
    print(f"Especificidade média: {resumo[melhor_config]['specificity']['mean']:.4f} ± {resumo[melhor_config]['specificity']['std']:.4f}")
    print("=" * 90)

    print("\n" + matriz_confusao_para_texto(resumo[melhor_config]["confusion_matrix"], int_to_class))

    salvar_relatorio(
        caminho_saida=args.saida,
        resumo=resumo,
        melhor_config=melhor_config,
        int_to_class=int_to_class,
        feature_names_final=feature_names_final,
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        n_folds=n_folds,
    )

    tempo_total = time.perf_counter() - inicio_total
    print(f"\n[INFO] Tempo total de execução: {tempo_total:.1f}s")
    print("Pipeline da classificação AV3 concluído com sucesso!")


if __name__ == "__main__":
    main()

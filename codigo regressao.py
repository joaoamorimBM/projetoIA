import numpy as np
import time

# =====================================================================
# 1. PRÉ-PROCESSAMENTO MANUAL (Sem Pandas/Scikit-Learn)
# =====================================================================

def carregar_dados_limpos(caminho_arquivo):
    """
    Lê o arquivo ARFF, mapeia valores textuais e exclui dados nulos (?) 
    para formar uma matriz estritamente numérica.
    """
    dados = []
    lendo_dados = False
    
    # Índices seguros (apenas números contínuos, sem '?') no dataset FPS Benchmark
    indices_seguros = [1, 2, 3, 8, 12, 17, 18, 19, 27, 35, 41]
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            
            # Pula cabeçalhos e comentários
            if not linha or linha.startswith('%') or linha.startswith('@'):
                if linha.upper().startswith('@DATA'):
                    lendo_dados = True
                continue
            
            if lendo_dados:
                valores = linha.split(',')
                linha_proc = []
                try:
                    # 1. Carrega as variáveis preditoras contínuas (X)
                    for i in indices_seguros:
                        linha_proc.append(float(valores[i]))
                    
                    # 2. Carrega e mapeia a qualidade gráfica (GameSetting)
                    cfg = valores[42].replace("'", "").replace('"', '').lower()
                    mapa = {'low': 1.0, 'med': 2.0, 'high': 3.0, 'max': 4.0}
                    linha_proc.append(mapa.get(cfg, 2.0))
                    
                    # 3. Carrega a Variável Alvo contínua: FPS (y)
                    linha_proc.append(float(valores[43]))
                    
                    dados.append(linha_proc)
                except ValueError:
                    # Ignora linhas defeituosas
                    continue

    matriz = np.array(dados)
    X = matriz[:, :-1] # Todas as colunas, exceto a última
    y = matriz[:, -1]  # Apenas a última coluna (FPS)
    return X, y

def normalizar_dados(X_treino, X_teste):
    """
    Aplica a padronização Z-score manualmente.
    A média e o desvio padrão são calculados APENAS no treino
    e aplicados no teste, para evitar o 'vazamento de dados' (data leakage).
    """
    medias = np.mean(X_treino, axis=0)
    desvios = np.std(X_treino, axis=0)
    
    desvios[desvios == 0] = 1.0 # Evita divisão por zero
    
    X_treino_norm = (X_treino - medias) / desvios
    X_teste_norm = (X_teste - medias) / desvios
    
    return X_treino_norm, X_teste_norm

def gerar_folds_manualmente(X, y, k=5):
    """
    Divide os dados em K-Folds para Validação Cruzada.
    """
    indices = np.arange(X.shape[0])
    np.random.seed(42) # Semente fixa para resultados reproduzíveis
    np.random.shuffle(indices)
    
    X_shuf = X[indices]
    y_shuf = y[indices]
    
    tamanho_fold = len(X) // k
    folds = []
    
    for i in range(k):
        inicio_teste = i * tamanho_fold
        fim_teste = (i + 1) * tamanho_fold if i != k - 1 else len(X)
        
        X_teste = X_shuf[inicio_teste:fim_teste]
        y_teste = y_shuf[inicio_teste:fim_teste]
        
        X_treino = np.concatenate((X_shuf[:inicio_teste], X_shuf[fim_teste:]), axis=0)
        y_treino = np.concatenate((y_shuf[:inicio_teste], y_shuf[fim_teste:]), axis=0)
        
        folds.append((X_treino, X_teste, y_treino, y_teste))
        
    return folds

# =====================================================================
# 2. ÁREA DOS ALGORITMOS DE MACHINE LEARNING
# =====================================================================

def adicionar_vies(X):
    """
    Adiciona uma coluna de '1's à matriz X para calcular o intercepto (beta 0).
    Isso permite que a reta de regressão não seja forçada a cruzar o zero.
    """
    coluna_uns = np.ones((X.shape[0], 1))
    return np.concatenate((coluna_uns, X), axis=1)

def treinar_regressao_multipla(X_treino, y_treino):
    """
    Implementação manual da Regressão Linear Múltipla usando a Equação Normal:
    beta = (X^T * X)^-1 * X^T * y
    """
    X_treino_b = adicionar_vies(X_treino)
    
    # 1. Calcula X Transposta
    X_T = X_treino_b.T
    
    # 2. Multiplica X Transposta por X
    X_T_X = X_T.dot(X_treino_b)
    
    # 3. Calcula a Inversa da matriz resultante
    try:
        inversa = np.linalg.inv(X_T_X)
    except np.linalg.LinAlgError:
        # Se a matriz for singular (difícil de inverter), usa a pseudo-inversa
        inversa = np.linalg.pinv(X_T_X)
        
    # 4. Multiplica a Inversa por X Transposta, e depois por y para achar os pesos (betas)
    betas = inversa.dot(X_T).dot(y_treino)
    
    return betas

def prever_regressao(X_teste, betas):
    """ Multiplica os dados de teste pelos pesos encontrados para prever o FPS """
    X_teste_b = adicionar_vies(X_teste)
    previsoes = X_teste_b.dot(betas)
    return previsoes

def calcular_r2(y_real, y_previsto):
    """
    Calcula a métrica R2-Score manualmente (Obrigatório pelo PDF).
    Fórmula: 1 - (Soma dos Resíduos Quadrados / Soma Total dos Quadrados)
    """
    media_y = np.mean(y_real)
    soma_residuos_quadrados = np.sum((y_real - y_previsto) ** 2)
    soma_total_quadrados = np.sum((y_real - media_y) ** 2)
    
    # Evita divisão por zero
    if soma_total_quadrados == 0:
        return 0.0
        
    r2 = 1 - (soma_residuos_quadrados / soma_total_quadrados)
    return r2

def calcular_r2_ajustado(r2, n, p):
    """
    Calcula o R² Ajustado.
    n = número de amostras (instâncias de teste)
    p = número de variáveis preditoras (atributos)
    """
    if n - p - 1 <= 0: # Evita divisão por zero ou negativa
        return 0.0
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def calcular_distancias(X_treino, x_teste_unico, tipo='euclidiana'):
    """
    Calcula a distância entre uma única amostra de teste e todas as amostras de treino.
    """
    if tipo == 'euclidiana':
        # Raiz quadrada da soma dos quadrados das diferenças
        distancias = np.sqrt(np.sum((X_treino - x_teste_unico) ** 2, axis=1))
    elif tipo == 'manhattan':
        # Soma do valor absoluto das diferenças
        distancias = np.sum(np.abs(X_treino - x_teste_unico), axis=1)
    return distancias

def prever_knn_regressao(X_treino, y_treino, X_teste, k=5, tipo_distancia='euclidiana'):
    """
    Prevê o valor (FPS) encontrando os K vizinhos mais próximos e fazendo a média dos seus valores.
    """
    previsoes = []
    
    # Para cada linha de teste, precisamos calcular a distância para TODO o treino
    for i in range(X_teste.shape[0]):
        # 1. Calcula as distâncias
        distancias = calcular_distancias(X_treino, X_teste[i], tipo=tipo_distancia)
        
        # 2. Pega os índices dos K menores valores (vizinhos mais próximos)
        indices_k_vizinhos = np.argsort(distancias)[:k]
        
        # 3. Pega os valores de FPS (y) desses vizinhos
        fps_vizinhos = y_treino[indices_k_vizinhos]
        
        # 4. A previsão é a média do FPS dos vizinhos
        previsoes.append(np.mean(fps_vizinhos))
        
    return np.array(previsoes)

# =====================================================================
# 3. EXECUÇÃO PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    print("--- Inteligência Artificial Computacional | UNIFOR ---")
    print("Iniciando o carregamento do dataset FPS Benchmark...\n")
    
    start_load = time.time()
    X, y = carregar_dados_limpos('file22f1639d20997.arff')
    end_load = time.time()
    
    print(f"Dados carregados com sucesso em {end_load - start_load:.2f} segundos!")
    print(f"Tamanho de X (Atributos): {X.shape[0]} amostras, {X.shape[1]} features.")
    print(f"Tamanho de y (Alvo - FPS): {y.shape[0]} amostras.\n")
    
    folds = gerar_folds_manualmente(X, y, k=5)
    
    # === VARIÁVEIS PARA ARMAZENAR RESULTADOS ===
    # Regressão Linear Múltipla
    r2_lr, r2_adj_lr, t_treino_lr, t_teste_lr = [], [], [], []
    
    # kNN Euclidiana
    r2_knn_euc, r2_adj_knn_euc, t_treino_knn_euc, t_teste_knn_euc = [], [], [], []
    
    # kNN Manhattan
    r2_knn_man, r2_adj_knn_man, t_treino_knn_man, t_teste_knn_man = [], [], [], []
    
    print("Iniciando o treinamento dos modelos nos 5 Folds...\n")
    
    for i, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
        print(f"--- Processando Fold {i+1} ---")
        X_tr_norm, X_te_norm = normalizar_dados(X_tr, X_te)
        
        n_amostras = len(y_te)
        p_features = X_tr_norm.shape[1]

        # ---------------------------------------------------------
        # 1. Regressão Linear Múltipla
        # ---------------------------------------------------------
        inicio_treino = time.time()
        betas = treinar_regressao_multipla(X_tr_norm, y_tr)
        fim_treino = time.time()
        
        inicio_teste = time.time()
        y_pred_lr = prever_regressao(X_te_norm, betas)
        fim_teste = time.time()
        
        r2 = calcular_r2(y_te, y_pred_lr)
        r2_adj = calcular_r2_ajustado(r2, n_amostras, p_features)
        
        r2_lr.append(r2)
        r2_adj_lr.append(r2_adj)
        t_treino_lr.append(fim_treino - inicio_treino)
        t_teste_lr.append(fim_teste - inicio_teste)

        # ---------------------------------------------------------
        # 2. kNN Euclidiana (k=5)
        # ---------------------------------------------------------
        inicio_treino = time.time()
        # Sem cálculo no treino
        fim_treino = time.time()
        
        inicio_teste = time.time()
        y_pred_knn_euc = prever_knn_regressao(X_tr_norm, y_tr, X_te_norm, k=5, tipo_distancia='euclidiana')
        fim_teste = time.time()
        
        r2 = calcular_r2(y_te, y_pred_knn_euc)
        r2_adj = calcular_r2_ajustado(r2, n_amostras, p_features)
        
        r2_knn_euc.append(r2)
        r2_adj_knn_euc.append(r2_adj)
        t_treino_knn_euc.append(fim_treino - inicio_treino)
        t_teste_knn_euc.append(fim_teste - inicio_teste)

        # ---------------------------------------------------------
        # 3. kNN Manhattan (k=5)
        # ---------------------------------------------------------
        inicio_treino = time.time()
        # Sem cálculo no treino
        fim_treino = time.time()
        
        inicio_teste = time.time()
        y_pred_knn_man = prever_knn_regressao(X_tr_norm, y_tr, X_te_norm, k=5, tipo_distancia='manhattan')
        fim_teste = time.time()
        
        r2 = calcular_r2(y_te, y_pred_knn_man)
        r2_adj = calcular_r2_ajustado(r2, n_amostras, p_features)
        
        r2_knn_man.append(r2)
        r2_adj_knn_man.append(r2_adj)
        t_treino_knn_man.append(fim_treino - inicio_treino)
        t_teste_knn_man.append(fim_teste - inicio_teste)


    print("\n=======================================================")
    print("   TABELA 1: RESULTADOS FINAIS DA REGRESSÃO")
    print("=======================================================")
    print(f"{'Algoritmo':<28} | {'R2-Score':<15} | {'R2 Ajustado':<15} | {'Tempo Treino (s)':<18} | {'Tempo Teste (s)':<18}")
    print("-" * 105)
    
    # Impressão formatada para Regressão Múltipla
    r2_str = f"{np.mean(r2_lr):.4f} ± {np.std(r2_lr):.4f}"
    r2_adj_str = f"{np.mean(r2_adj_lr):.4f} ± {np.std(r2_adj_lr):.4f}"
    t_treino_str = f"{np.mean(t_treino_lr):.5f} ± {np.std(t_treino_lr):.5f}"
    t_teste_str = f"{np.mean(t_teste_lr):.5f} ± {np.std(t_teste_lr):.5f}"
    print(f"{'Regressão Linear Múltipla':<28} | {r2_str:<15} | {r2_adj_str:<15} | {t_treino_str:<18} | {t_teste_str:<18}")

    # Impressão formatada para kNN Euclidiana
    r2_str = f"{np.mean(r2_knn_euc):.4f} ± {np.std(r2_knn_euc):.4f}"
    r2_adj_str = f"{np.mean(r2_adj_knn_euc):.4f} ± {np.std(r2_adj_knn_euc):.4f}"
    t_treino_str = f"{np.mean(t_treino_knn_euc):.5f} ± {np.std(t_treino_knn_euc):.5f}"
    t_teste_str = f"{np.mean(t_teste_knn_euc):.5f} ± {np.std(t_teste_knn_euc):.5f}"
    print(f"{'kNN (Dist. Euclidiana)':<28} | {r2_str:<15} | {r2_adj_str:<15} | {t_treino_str:<18} | {t_teste_str:<18}")
    
    # Impressão formatada para kNN Manhattan
    r2_str = f"{np.mean(r2_knn_man):.4f} ± {np.std(r2_knn_man):.4f}"
    r2_adj_str = f"{np.mean(r2_adj_knn_man):.4f} ± {np.std(r2_adj_knn_man):.4f}"
    t_treino_str = f"{np.mean(t_treino_knn_man):.5f} ± {np.std(t_treino_knn_man):.5f}"
    t_teste_str = f"{np.mean(t_teste_knn_man):.5f} ± {np.std(t_teste_knn_man):.5f}"
    print(f"{'kNN (Dist. Manhattan)':<28} | {r2_str:<15} | {r2_adj_str:<15} | {t_treino_str:<18} | {t_teste_str:<18}")
    print("=======================================================\n")

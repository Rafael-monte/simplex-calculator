import resources.calc as calculos
import resources.matrix_handling as matrizes
import numpy as np

iteracoes=0
solucao_otima_encontrada = False
def apply_simplex(coeficientes_equacao_minimizacao, matriz_A, indices_base, indices_nao_base, matriz_resultado):
    # Fase I: Atualização
    print(f'-----------------------Atualização-----------------------')
    matriz_base = matrizes.pick_matrix_by_index_columns(matriz_A, indices_base)
    matriz_nao_base = matrizes.pick_matrix_by_index_columns(matriz_A, indices_nao_base)
    constantes_base = matrizes.get_constants_by_indexes(coeficientes_equacao_minimizacao, indices_base)
    constantes_nao_base = matrizes.get_constants_by_indexes(coeficientes_equacao_minimizacao, indices_nao_base)
    print(f'Indices Base = {indices_base}')
    print(f'Indices não base = {indices_nao_base}')
    print(f'B={matriz_base}')
    print(f'N={matriz_nao_base}')
    print(f'Cbt={constantes_base}')
    print(f'Cnt={constantes_nao_base}')
    global iteracoes
    global solucao_otima_encontrada
    # Iniciando iteracao
    iteracoes+=1
    print(f'-----------------------{iteracoes}a Iteração-----------------------')

    # Passo 1: Calcular Solução básica
    solucao_basica = matrizes.get_basic_solution(matriz_base, matriz_resultado)
    print(f'Solução Básica={solucao_basica}')
    # Passo 2i: Calcular Multiplicador simplex
    multiplicador_simplex = matrizes.get_λt(matriz_base, constantes_base)
    print(f'λt = {multiplicador_simplex}')
    # Passo 2ii: Calcular Custos Parciais
    custos_parciais = matrizes.get_parcial_weights_by_non_base_non_base_constants_and_simplex_multiplier(matriz_nao_base, constantes_nao_base, multiplicador_simplex)
    print(f'Cnjs={custos_parciais}')
    # Passo 2iii: Calcular Cnk e k
    (cnk, k) = calculos.get_Cnk_and_k(custos_parciais)
    print(f'Cnk = {cnk}\nk={k + 1}, logo N{k + 1} entra na base')
    # Passo 3: Cnk >= 0?
    if (cnk >= 0): solucao_otima_encontrada = True
    # Passo 4: Y
    y = matrizes.get_y(matriz_base, matriz_nao_base, k)
    print(f'Y = {y}')
    # Passo 5: Y <= 0?
    if matrizes.is_matrix_less_or_equal_empty(y): raise Exception("O problema não possui solução pois Y <= 0")
    # Epsilon
    (epsilon, indice_saida) = calculos.get_Ɛ_and_basis_outgoing_index(solucao_basica, y)
    print(f'Ɛ={epsilon}, logo B{indice_saida + 1} sai da base')

    if iteracoes > solucao_basica.shape[0] + 1:
        raise Exception('O problema não possui solucao')

    if not solucao_otima_encontrada:
        # Troca de indices
        buffer=indices_base[indice_saida]
        indices_base[indice_saida]=indices_nao_base[k]
        indices_nao_base[k]=buffer
        apply_simplex(
            coeficientes_equacao_minimizacao=coeficientes_equacao_minimizacao,
            matriz_A=matriz_A,
            indices_base=indices_base,
            indices_nao_base=indices_nao_base,
            matriz_resultado=matriz_resultado
        )
    else:
        solucao_final= f"[{''.join(f'X{i+1}, ' for i in range(len(coeficientes_equacao_minimizacao)))}]"
        indice_ultima_virgula=solucao_final.rfind(", ")
        if indice_ultima_virgula != -1:
            solucao_final = solucao_final[:indice_ultima_virgula] + solucao_final[indice_ultima_virgula+2:]
        valores_solucao=[0 for _ in range(len(coeficientes_equacao_minimizacao))]
        count=0
        for (indice_linha, indice_base) in enumerate(indices_base):
            valores_solucao[indice_base - 1]=solucao_basica[indice_linha][0]
        print(f'Solução final = {solucao_final} = {valores_solucao}')
        return solucao_basica

# Definições
# Matriz A
matriz_A=np.array([[1, 2, 3, 1, 0], [3, 2, 3, 0, 1]])

# Coeficientes da equação de minimização
coeficientes_equacao_minimizacao=np.array([-1, -9, -3, 0, 0])

# Indices
indices_base=np.array([1, 2])
indices_nao_base=np.array([3, 4, 5])

# Matriz Resultado (b)
matriz_resultado=np.array([[9], [15]])

apply_simplex(coeficientes_equacao_minimizacao=coeficientes_equacao_minimizacao,
              matriz_A=matriz_A,
              indices_base=indices_base,
              indices_nao_base=indices_nao_base,
              matriz_resultado=matriz_resultado
              )
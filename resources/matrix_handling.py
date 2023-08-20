import numpy as np

def get_λt(base_matrix, base_constants):
    """
    Retorna o multiplicador simplex, dadas a matriz base e as constantes da base. \n
    Equação:\n\tλt =Cbt * B^-1
    """
    base_inverse = np.linalg.inv(base_matrix)
    λt = np.dot(base_constants, base_inverse)
    return λt

def get_basic_solution(base_matrix, results_matrix):
    """
    Retorna a solução básica da iteração. \n
    Equação:\n\tX̄b = B^-1 * b
    """
    base_inverse = np.linalg.inv(base_matrix)
    basic_solution = np.dot(base_inverse, results_matrix)
    return basic_solution

def get_y(base_matrix, non_base_matrix, k):
    """
    Retorna Y dadas as matrizes base e não base, junto ao fator k. \n
    Equação:\n\tY = B^-1 an_(k)
    """
    k-=1
    base_inverse = np.linalg.inv(base_matrix)
    k_column=transpose(non_base_matrix[:, k])
    y=np.dot(base_inverse, k_column)
    return y

def transpose(line_vector):
    """
    Transpõe um vetor linha
    """
    return np.array(line_vector).reshape(-1, 1) 
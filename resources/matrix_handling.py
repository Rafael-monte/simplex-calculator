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

def pick_base_matrix_by_index_columns(A_matrix, base_columns_indexes: set[int]):
    """
    Retorna a matriz base ao informar a matriz A e os indices da base
    """
    A_matrix=np.array(A_matrix)
    base_matrix=np.array([[] for i in base_columns_indexes])
    for column_index in base_columns_indexes:
        column_index-=1
        column=transpose(A_matrix[:, column_index])
        base_matrix = np.hstack((base_matrix, column))
    return base_matrix

def transpose(line_vector):
    """
    Transpõe um vetor linha
    """
    return np.array(line_vector).reshape(-1, 1)

def get_parcial_weights_by_non_base_non_base_constants_and_simplex_multiplier(non_base, non_base_constants, λt) -> list[float]:
	"""
	Calcula os pesos parciais, dados a não-base, constantes da não-base e multiplicador simplex\n
	\tEquação:\n
	\tĈnj = Cnj - λt * anj, j={1,2,..., n} 
	"""
	parcial_weights=[]
	for i in range(len(non_base_constants)):
		cnj=non_base_constants[i]
		anj=transpose(non_base[:, i])
		λtanj=np.dot(λt, anj)
		parcial_weight=(cnj - λtanj)[0]
		parcial_weights.append(parcial_weight)
	return parcial_weights
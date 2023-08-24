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
    base_inverse = np.linalg.inv(base_matrix)
    k_column=transpose(non_base_matrix[:, k])
    y=np.dot(base_inverse, k_column)
    return y

def pick_matrix_by_index_columns(A_matrix, columns_indexes: set[int]):
    """
    Retorna uma submatriz ao informar a matriz A e os indices dos vetores
    """
    A_matrix=np.array(A_matrix)
    matrix=np.array([[] for _ in range(A_matrix.shape[0])])
    for column_index in columns_indexes:
        column_index-=1 # Índice normalizado
        column=transpose(A_matrix[:, column_index])
        matrix = np.hstack((matrix, column))
    return matrix

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

def get_constants_by_indexes(minimization_equation_constants: list[float], indexes: list[int]) -> list[float]:
     """
     Busca as constantes da equação de minimização dados os indices
     """
     constants=[]
     for index in indexes:
        index-=1 # Índice normalizado
        for (i_equation_constant, equation_constant) in enumerate(minimization_equation_constants):
             if i_equation_constant == index:
                  constants.append(equation_constant)
     return constants


def is_matrix_less_or_equal_empty(y)->bool:
    num_columns=y.shape[0]
    for column_index in range(num_columns):
         if y[column_index][0] > 0:
              return False
    return True
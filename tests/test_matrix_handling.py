import unittest
import numpy as np
import resources.matrix_handling as matrixes
class TestMatrixHandling(unittest.TestCase):

    def setUp(self) -> None:
       self.matrizes_base = [
           np.identity(3),
           np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]]),
           np.array([[2, 0, -1], [3, 1, 1], [0, 0, 1]])
       ]
       self.constantes_base=[
           np.array([0, 0, 0]),
           np.array([0, 0, -5]),
           np.array([-1, 0, -5])
       ]
       self.solucoes_base=[
           np.array([[6], [12], [4]]),
           np.array([[10], [8], [4]]),
           np.array([[5], [-7], [4]])
       ]
       self.multiplicadores=[
           np.array([0,0,0]),
           np.array([0,0,-5]),
           np.array([-1/2, 0, -11/2])
       ]
       self.nao_bases=[
           np.array([[1, 2, 4, -1], [2, 3, -1, 1], [1, 0, 1, 1]]),
           np.array([[1, 2, 4, 0], [2, 3, -1, 0], [1, 0, 1, 1]])
       ]
       self.ys=[
           np.array([[-1], [1], [1]]),
           np.array([[2],[3],[0]])
       ]
       self.ks=[
           4, 2
       ]
       self.matriz_A=[
           [1, 2, 4, -1, 1, 0, 0],
           [2, 3, -1, 1, 0, 1, 0],
           [1, 0, 1,  1, 0, 0, 1]
       ]
       self.indices_base={5,6,7}
       self.pesos_parciais=[
            [-2, -1, 3, -5],
            [3, -1, 8, 5]
       ]
       self.constantes_nao_base=[
           [-2, -1, 3, -5],
           [-2, -1, 3, 0]
       ]
       self.constantes_equacao_minimizacao=[
           [-1, -9, -3, 0, 0]
       ]

    def test_deve_retornar_multiplicador_simplex_ao_informar_base_e_vetor_constantes_da_base(self):
        for i in range(len(self.multiplicadores)):
            base_matrix=self.matrizes_base[i]
            base_constants=self.constantes_base[i]
            λt = matrixes.get_λt(base_matrix=base_matrix, base_constants=base_constants)
            multiplicador_correto=self.multiplicadores[i]
            self.assertTrue(np.array_equal(λt, multiplicador_correto))

    def test_deve_retornar_solucoes_da_base_ao_informar_base_e_matriz_solucao(self):
        for i in range(len(self.solucoes_base)):
            base_matrix=self.matrizes_base[i]
            results_matrix=np.array([[6], [12], [4]])
            basic_solution=matrixes.get_basic_solution(base_matrix=base_matrix, results_matrix=results_matrix)
            self.assertTrue(np.array_equal(basic_solution, self.solucoes_base[i]))

    def test_deve_retornar_y_ao_informar_base_nao_base_e_fator_k(self):
        for i in range(len(self.ys)):
            base_matrix=self.matrizes_base[i]
            k=self.ks[i] - 1 # Indice normalizado
            non_base=self.nao_bases[i]
            y=matrixes.get_y(base_matrix=base_matrix, non_base_matrix=non_base, k=k)
            self.assertTrue(np.array_equal(y, self.ys[i]))

    def test_deve_retornar_a_matriz_base_ao_informar_matriz_A_e_indices_base(self):
        matriz_identidade=self.matrizes_base[0]
        matriz_base=matrixes.pick_matrix_by_index_columns(self.matriz_A, self.indices_base)
        self.assertTrue(np.array_equal(matriz_identidade, matriz_base))

    def test_deve_retornar_pesos_parciais_ao_informar_matriz_nao_base_constantes_nao_base_e_multiplicador_simplex(self):
        for i in range(len(self.pesos_parciais)):
            nao_base=self.nao_bases[i]
            constantes_nao_base=self.constantes_nao_base[i]
            λt=self.multiplicadores[i]
            pesos_parciais=matrixes.get_parcial_weights_by_non_base_non_base_constants_and_simplex_multiplier(nao_base, constantes_nao_base, λt)
            self.assertTrue(np.array_equal(pesos_parciais, self.pesos_parciais[i]))

    def test_deve_retornar_coeficientes_base_ao_informar_indices_e_constantes_equacao(self):
        indices_base=[1, 2]
        constantes_equacao_minimizacao=self.constantes_equacao_minimizacao[0]
        constantes_base=matrixes.get_constants_by_indexes(constantes_equacao_minimizacao, indices_base)
        self.assertTrue(np.array_equal([-1, -9], constantes_base))
        indices_nao_base=[3,4,5]
        constantes_nao_base=matrixes.get_constants_by_indexes(constantes_equacao_minimizacao, indices_nao_base)
        self.assertTrue(np.array_equal(constantes_nao_base, [-3, 0, 0]))
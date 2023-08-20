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
            k=self.ks[i]
            non_base=self.nao_bases[i]
            y=matrixes.get_y(base_matrix=base_matrix, non_base_matrix=non_base, k=k)
            self.assertTrue(np.array_equal(y, self.ys[i]))
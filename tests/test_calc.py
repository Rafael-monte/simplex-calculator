import unittest
import resources.calc as calculos
import numpy as np
class TestCalc(unittest.TestCase):
    def setUp(self) -> None:
        self.pesos_parciais=[
            [1.50, -2, -59, 0],
            [0.333, 20, 5, 2],
            [0, 57, 33, 28, 4]
        ]
        self.cnks_e_indices=[(-59, 2), (0.333, 0), (0, 0)]
        self.matrizes_x=[
            np.array([[2], [6], [4]]),
            np.array([[4], [6], [2]])
        ]
        self.ys=[
            np.array([[-1], [-2], [1]]),
            np.array([[2], [0], [-1]])
        ]
        self.Ɛs_e_indices_de_saida=[
            (4, 2),
            (2, 0)
        ]
    
    def test_deve_retornar_menor_valor_e_indice_cnk_ao_informar_pesos_parciais(self):
        for i in range(len(self.cnks_e_indices)):
            pesos=self.pesos_parciais[i]
            cnk_e_indice= calculos.get_Cnk_and_k(pesos)
            self.assertEquals(cnk_e_indice, self.cnks_e_indices[i])

    def test_deve_retornar_Ɛ_ao_informar_x_base_e_y(self):
        for i in range(len(self.matrizes_x)):
            x_matrix=self.matrizes_x[i]
            y=self.ys[i]
            Ɛ_e_indice_saida=calculos.get_Ɛ_and_basis_outgoing_index(x_matrix, y)
            self.assertEquals(Ɛ_e_indice_saida, self.Ɛs_e_indices_de_saida[i])

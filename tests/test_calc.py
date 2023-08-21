import unittest
import resources.calc as calculos
class TestCalc(unittest.TestCase):
    def setUp(self) -> None:
        self.pesos_parciais=[
            [1.50, -2, -59, 0],
            [0.333, 20, 5, 2],
            [0, 57, 33, 28, 4]
        ]
        self.cnks_e_indices=[(-59, 2), (0.333, 0), (0, 0)]
    
    def test_deve_retornar_menor_valor_e_indice_cnk_ao_informar_pesos_parciais(self):
        for i in range(len(self.cnks_e_indices)):
            pesos=self.pesos_parciais[i]
            cnk_e_indice= calculos.get_Cnk_and_k(pesos)
            self.assertEquals(cnk_e_indice, self.cnks_e_indices[i])
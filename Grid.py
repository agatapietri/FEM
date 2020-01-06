from Element import Element
from Node import Node
from numpy import arange
from UniversalElement import UniversalElement
import numpy as np
class Grid:
    def __init__(self, H, W, nH, nW):
        self.nodes = []
        self.elements = []
        self.nN = nH * nW  #ilosc wezlow
        self.nE = (nH - 1) * (nW - 1) #ilosc elementow

        elementWidth = W / (nW - 1)
        elementHeight = H / (nH - 1)

        node_id = 1

        for x in arange(0, W + elementWidth, elementWidth):
            for y in arange(0, H + elementHeight, elementHeight):
                new_node = Node(x, y, None, None, node_id)
                if x == 0 or x == W or y == 0 or y ==H:
                    new_node.boundary_condition = True
                else:
                    new_node.boundary_condition = False

                self.nodes.append(new_node)
                node_id += 1

        element_id = 1
        for x in arange(0, nW - 1):
            for y in arange(0, nH - 1):
                left_bottom = x * nH + y
                left_top = left_bottom + 1
                right_bottom = (x + 1) * nH + y
                right_top = right_bottom + 1

                element_nodes_tab = [ self.nodes[left_bottom], self.nodes[right_bottom],
                                      self.nodes[right_top], self.nodes[left_top]]

                new_element = Element(element_nodes_tab, element_id)
                self.elements.append(new_element)
                element_id += 1

    def H_matrix_global(self, universal_element, conductivity):

        global_h = np.zeros(shape=(self.nN, self.nN))
        for i in range (self.nE):
            element = self.elements[i]
            H_local = universal_element.H_matrix(universal_element, conductivity)
            element_index = [element[i].id for i in range(4)]

            for y in range(4):
                for x in range(4):
                    global_h[element_index[x], element_index[y]] += H_local[x, y]

        return global_h

    def _Cmatrix_global(self,  universal_element, density, specific_heat):

        global_c = np.zeros(shape=(self.nN, self.nN))

        for i in range(self.nE):
            element = self.elements[i]
            C_local = universal_element.H_matrix(universal_element, density, specific_heat)
            element_index = [element[i].id for i in range(4)]

            for y in range(4):
                for x in range(4):
                    global_c[element_index[x], element_index[y]] += C_local[x, y]

        return global_c




if __name__ == "__main__":
    test = Grid(0.1, 0.1, 4, 4)
    H_global_test = test.H_matrix_global(UniversalElement( ), 30)
    print(H_global_test)
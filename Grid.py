from Element import Element
from Node import Node
from numpy import arange
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
                element_id += 1

if __name__ == "__main__":
    test = Grid(3, 3, 4, 4)
    print(test)
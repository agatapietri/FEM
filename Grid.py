from Element import Element
from Node import Node
from numpy import arange
from UniversalElement import UniversalElement
import numpy as np
import json


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
        for i in range(self.nE):
            element = self.elements[i]
            H_local = element.H_matrix(universal_element, conductivity)
            element_index = [element.nodes[i].id for i in range(4)]

            for y in range(4):
                for x in range(4):
                    global_h[element_index[x] - 1, element_index[y] - 1] += H_local[x, y]

        return global_h

    def C_matrix_global(self,  universal_element, density, specific_heat):

        global_c = np.zeros(shape=(self.nN, self.nN))

        for i in range(self.nE):
            element = self.elements[i]
            C_local = element.C_matrix(universal_element, density, specific_heat)
            element_index = [element.nodes[i].id for i in range(4)]

            for y in range(4):
                for x in range(4):
                    global_c[element_index[x] - 1, element_index[y] - 1] += C_local[x, y]

        return global_c

    def H_BC_global(self,universal_element, alpha):
        global_H_BC = np.zeros(shape=(self.nN, self.nN))

        for i in range(self.nE):
            element = self.elements[i]
            H_BC_local = element.H_matrix_BC(universal_element, alpha)
            element_index = [element.nodes[i].id for i in range(4)]

            for y in range(4):
                for x in range(4):
                    global_H_BC[element_index[x] - 1, element_index[y] - 1] += H_BC_local[x, y]

        return global_H_BC

    def P_vector_global(self, universal_element, alpha, temp):
        global_P_vector = np.zeros(shape=(self.nN, 1))

        for i in range(self.nE):
            element = self.elements[i]
            P_vector_local = element.P_vector(universal_element, alpha, temp)
            element_index = [element.nodes[i].id for i in range(4)]
            for x in range(4):
                global_P_vector[element_index[x] - 1, 0] += P_vector_local[x, 0]

        return global_P_vector

    def simulation(self, universal_element, initial_temp, simulation_time, sim_time_step,ambient_temp,
                   alpha, specific_heat, conductivity, density):
        t0_Vector = [initial_temp for i in range(self.nN)]
        t0_Vector = np.asanyarray([t0_Vector]).reshape((self.nN,1))
        step_amount = (int)(simulation_time/sim_time_step)
        step_in_time = [(step + 1) * sim_time_step for step in range(step_amount)]

        for actual_time in step_in_time:

            global_h = self.H_matrix_global(universal_element, conductivity)
            global_c = self.C_matrix_global(universal_element, density, specific_heat)
            p_vector_global = self.P_vector_global(universal_element, alpha, ambient_temp)
            global_h_bc = self.H_BC_global(universal_element, alpha)
            global_h += global_h_bc
            global_c *= 1/sim_time_step
            p_vector_global *= -1
            p_vector_global += np.dot(global_c, t0_Vector)
            global_h += global_c

            t1 = np.linalg.solve(global_h, p_vector_global)
            t1_max = np.amax(t1)
            t1_min = np.amin(t1)

            print("Step: = ", actual_time, " min: = ", t1_min, " max: = ", t1_max)
            t0_Vector = t1

        return t0_Vector


if __name__ == "__main__":

    with open('MES1.json', 'r') as data_file:
        mes = json.load(data_file)
    test = Grid(mes["H"], mes["W"], mes["N_H"], mes["N_W"])
    P_vector_glogal = test.P_vector_global(UniversalElement(), 300, 1200)
    print(P_vector_glogal)
    print("------------------------------------------------------")
    H_global_test = test.H_matrix_global(UniversalElement( ), 30)
    print(H_global_test)
    C_global_test = test.C_matrix_global(UniversalElement( ), 7800, 700)
    print(C_global_test)
    print("------------------------------------------------------")
    H_BC_global_test = test.H_BC_global(UniversalElement( ), 25)
    print(H_global_test)
    print("------------------------------------------------------")
    P_vector_test = test.P_vector_global(UniversalElement( ), 25, 1000)
    print(P_vector_test)
    print("------------------------------------------------------")
    simulation_test = test.simulation(UniversalElement(),mes["initial_temp"], mes["simulation_time"],
                                      mes["step_time"], mes["ambient_temp"], mes["alpha"],
                                      mes["specific_heat"], mes["conductivity"], mes["density"])
    print(simulation_test)
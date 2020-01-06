import Node
from UniversalElement import UniversalElement
import numpy as np
from IntegralPoint import IntegralPoint
import math

class Element:
    def __init__(self, nodes: [Node], id):
        self.nodes = nodes
        self.id = id

    def jacobian(self, universal_element: UniversalElement, integral_point_id):
        dx_dEta = 0.
        dx_dKsi = 0.
        dy_dEta = 0.
        dy_dKsi = 0.

        for i in range(4):
            dx_dKsi += universal_element.N_ksi_matrix[integral_point_id, i] * self.nodes[i].x
            dx_dEta += universal_element.N_eta_matrix[integral_point_id, i] * self.nodes[i].x
            dy_dKsi += universal_element.N_ksi_matrix[integral_point_id, i] * self.nodes[i].y
            dy_dEta += universal_element.N_eta_matrix[integral_point_id, i] * self.nodes[i].y

        jacobian = np.zeros(shape=(2, 2))
        jacobian[0, 0] = dx_dKsi
        jacobian[0, 1] = dy_dKsi
        jacobian[1, 0] = dx_dEta
        jacobian[1, 1] = dy_dEta

        return jacobian

    def dN_dx_vector(self, universal_element, integral_point_id):

        dN_dX = np.zeros(shape=(4, 1))

        jacobian = self.jacobian(universal_element, integral_point_id)
        inversed_jacobian = np.linalg.inv(jacobian)

        for i in range(4):
            dNi_dX = inversed_jacobian[0, 0] * universal_element.N_ksi_matrix[integral_point_id, i] \
                     + inversed_jacobian[0, 1] * universal_element.N_eta_matrix[integral_point_id, i]

            dN_dX[i, 0] = dNi_dX

        return dN_dX

    def dN_dY_vector(self, universal_element, integral_point_id):

        dN_dY = np.zeros(shape=(4, 1))

        jacobian = self.jacobian(universal_element, integral_point_id)
        inversed_jacobian = np.linalg.inv(jacobian)

        for i in range(4):
            dNi_dY = inversed_jacobian[1, 0] * universal_element.N_ksi_matrix[integral_point_id, i] +\
                     inversed_jacobian[1, 1] * universal_element.N_eta_matrix[integral_point_id, i]

            dN_dY[i, 0] = dNi_dY

        return dN_dY

    def H_matrix(self, universal_element, conductivity):

        H_matrix_for_h_matrix_ineg_point = []

        for integral_point_id in range(4):
            dN_deX_vector = self.dN_dx_vector(universal_element, integral_point_id)
            dN_deY_vector = self.dN_dY_vector(universal_element, integral_point_id)
            dN_deX_vector_transp = np.transpose(dN_deX_vector)
            dN_deY_vector_transp = np.transpose(dN_deY_vector)

            h_matrix_in_iteg_point = np.dot(dN_deX_vector, dN_deX_vector_transp) + np.dot(dN_deY_vector, dN_deY_vector_transp)
            h_matrix_in_iteg_point *= conductivity
            H_matrix_for_h_matrix_ineg_point.append(h_matrix_in_iteg_point)

        H_matrix = np.zeros(shape=(4, 4))
        for integral_point_id in range(4):
            h_matrix_in_iteg_point = H_matrix_for_h_matrix_ineg_point[integral_point_id]
            weight1 = universal_element.integral_points[integral_point_id].weight_ksi
            weight2 = universal_element.integral_points[integral_point_id].weight_eta
            det_J = np.linalg.det(self.jacobian(universal_element, integral_point_id))

            H_matrix += h_matrix_in_iteg_point * weight1 * weight2 * det_J

        return H_matrix


    def C_matrix(self, universal_element, density, specific_heat):

        C_matrix_for_c_matrix_integ_point = []
        for integral_point_id in range(4):
            N_vector = universal_element.N_matrix[integral_point_id]
            N_vector = np.asarray([N_vector])
            N_vector = N_vector.reshape((4, 1))

            N_vector_transp = np.transpose(N_vector)

            c_matrix_in_iteg_point = np.dot(N_vector, N_vector_transp)
            c_matrix_in_iteg_point *= density * specific_heat

            C_matrix_for_c_matrix_integ_point.append(c_matrix_in_iteg_point)

        C_matrix = np.zeros(shape=(4, 4))
        for integral_point_id in range(4):
            c_matrix_in_iteg_point = C_matrix_for_c_matrix_integ_point[integral_point_id]
            weight1 = universal_element.integral_points[integral_point_id].weight_ksi
            weight2 = universal_element.integral_points[integral_point_id].weight_eta
            det_J = np.linalg.det(self.jacobian(universal_element, integral_point_id))

            C_matrix += c_matrix_in_iteg_point * weight1 * weight2 * det_J

        return C_matrix

    def H_matrix_BC(self, universal_element, alpha):

        integral_points_for_side = []
        for i in range(4):
            integral_points_for_side.append([1,1])
        integral_points_for_side[0][0] = IntegralPoint(-1 / math.sqrt(3), -1, 1, 1)
        integral_points_for_side[0][1] = IntegralPoint(1 / math.sqrt(3), -1, 1, 1)

        integral_points_for_side[1][0] = IntegralPoint(1, -1 / math.sqrt(3), 1, 1)
        integral_points_for_side[1][1] = IntegralPoint(1, 1 / math.sqrt(3), 1, 1)

        integral_points_for_side[2][0] = IntegralPoint(1 / math.sqrt(3), 1, 1, 1)
        integral_points_for_side[2][1] = IntegralPoint(-1 / math.sqrt(3), 1, 1, 1)

        integral_points_for_side[3][0] = IntegralPoint(-1, 1 / math.sqrt(3), 1, 1)
        integral_points_for_side[3][1] = IntegralPoint(-1, -1 / math.sqrt(3), 1, 1)

        H_BC_matrix = np.zeros(shape=(4,4))
        for i in range(4):
            edge_with_boundary_cond = None

            if self.nodes[i].boundary_condition and self.nodes[(i+1) % 4]:
                edge_with_boundary_cond = (self.nodes[i], self.nodes[(i+1) % 4])
                N_vector_for_iteg_point1 = universal_element.N_vector(integral_points_for_side[i][0])
                N_vector_for_iteg_point_transp1 = np.transpose(N_vector_for_iteg_point1)
                N_matrix_for_integ_point1 = np.dot(N_vector_for_iteg_point1, N_vector_for_iteg_point_transp1)
                N_vector_for_iteg_point2 = universal_element.N_vector(integral_points_for_side[i][1])
                N_vector_for_iteg_point_transp2 = np.transpose(N_vector_for_iteg_point2)
                N_matrix_for_integ_point2 = np.dot(N_vector_for_iteg_point2, N_vector_for_iteg_point_transp2)
                distance_between_nodes = math.sqrt((self.nodes[i].x - self.nodes[(i+1) % 4].x) ** 2
                                                   + (self.nodes[i].y - self.nodes[(i+1) % 4].y) ** 2)
                jacobian1D = distance_between_nodes * 0.5
                H_BC_matrix_for_side = N_matrix_for_integ_point1 * jacobian1D + N_matrix_for_integ_point2 * jacobian1D
                H_BC_matrix_for_side *= alpha
            else:
                continue

            H_BC_matrix += H_BC_matrix_for_side

        return H_BC_matrix

    def P_vector (self, universal_element, alpha, temp):
        integral_points_for_side = []
        for i in range(4):
            integral_points_for_side.append([1, 1])
        integral_points_for_side[0][0] = IntegralPoint(-1 / math.sqrt(3), -1, 1, 1)
        integral_points_for_side[0][1] = IntegralPoint(1 / math.sqrt(3), -1, 1, 1)

        integral_points_for_side[1][0] = IntegralPoint(1, -1 / math.sqrt(3), 1, 1)
        integral_points_for_side[1][1] = IntegralPoint(1, 1 / math.sqrt(3), 1, 1)

        integral_points_for_side[2][0] = IntegralPoint(1 / math.sqrt(3), 1, 1, 1)
        integral_points_for_side[2][1] = IntegralPoint(-1 / math.sqrt(3), 1, 1, 1)

        integral_points_for_side[3][0] = IntegralPoint(-1, 1 / math.sqrt(3), 1, 1)
        integral_points_for_side[3][1] = IntegralPoint(-1, -1 / math.sqrt(3), 1, 1)
        P_vector = np.zeros(shape=(4, 1))

        for i in range(4):
            edge_with_boundary_cond = None

            if self.nodes[i].boundary_condition and self.nodes[(i + 1) % 4]:
                edge_with_boundary_cond = (self.nodes[i], self.nodes[(i + 1) % 4])
                N_vector_for_iteg_point1 = universal_element.N_vector(integral_points_for_side[i][0])

                N_vector_for_iteg_point2 = universal_element.N_vector(integral_points_for_side[i][1])

                distance_between_nodes = math.sqrt((self.nodes[i].x - self.nodes[(i + 1) % 4].x) ** 2
                                                   + (self.nodes[i].y - self.nodes[(i + 1) % 4].y) ** 2)
                jacobian1D = distance_between_nodes * 0.5
                P_vector_for_side = N_vector_for_iteg_point1 * jacobian1D + N_vector_for_iteg_point1 * jacobian1D
                P_vector_for_side *= alpha * temp
            else:
                continue

            P_vector += P_vector_for_side

        return -P_vector







if __name__ == "__main__":
    node1 = Node.Node(0., 0., True, None, 0)
    node2 = Node.Node(0.025, 0., True, None, 2)
    node3 = Node.Node(0.025, 0.025, True, None, 3)
    node4 = Node.Node(0., 0.025, True, None, 1)
    test_element = Element([node1, node2, node3, node4], 0)
    h_matrix_in_element = test_element.H_matrix(UniversalElement( ), 30)
    print(h_matrix_in_element)
    h_bc_matrix_in_element = test_element.H_matrix_BC(UniversalElement( ), 25)
    print(h_bc_matrix_in_element)
    c_matrix_in_element = test_element.C_matrix(UniversalElement( ), 7800, 700)
    print(c_matrix_in_element)
    p_vector_in_element = test_element.P_vector(UniversalElement(),25, 1000)
    print(p_vector_in_element)



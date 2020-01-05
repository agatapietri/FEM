from IntegralPoint import IntegralPoint
import numpy as np
import ShapeFunctions
import math


class UniversalElement:

    def __init__(self):
        self.integral_points = []
        self.N_matrix = np.zeros(shape=(4, 4))
        self.N_ksi_matrix = np.zeros(shape=(4, 4))
        self.N_eta_matrix = np.zeros(shape=(4, 4))

        self.integral_points.append(IntegralPoint(-1/math.sqrt(3), -1/math.sqrt(3), 1, 1))
        self.integral_points.append(IntegralPoint(1 / math.sqrt(3), -1 / math.sqrt(3), 1, 1))
        self.integral_points.append(IntegralPoint(1 / math.sqrt(3), 1 / math.sqrt(3), 1, 1))
        self.integral_points.append(IntegralPoint(-1 / math.sqrt(3), 1 / math.sqrt(3), 1, 1))

        i = 0
        for integral_point in self.integral_points:
            self.N_matrix[i, 0] = ShapeFunctions.first_shape_function(integral_point)
            self.N_matrix[i, 1] = ShapeFunctions.second_shape_function(integral_point)
            self.N_matrix[i, 2] = ShapeFunctions.third_shape_function(integral_point)
            self.N_matrix[i, 3] = ShapeFunctions.fourth_shape_function(integral_point)
            i += 1

        i = 0
        for integral_point in self.integral_points:
            self.N_eta_matrix[i, 0] = ShapeFunctions.first_shape_function_deta(integral_point)
            self.N_eta_matrix[i, 1] = ShapeFunctions.second_shape_function_deta(integral_point)
            self.N_eta_matrix[i, 2] = ShapeFunctions.third_shape_function_deta(integral_point)
            self.N_eta_matrix[i, 3] = ShapeFunctions.fourth_shape_function_deta(integral_point)
            i += 1

        i = 0
        for integral_point in self.integral_points:
            self.N_ksi_matrix[i, 0] = ShapeFunctions.first_shape_function_dksi(integral_point)
            self.N_ksi_matrix[i, 1] = ShapeFunctions.second_shape_function_dksi(integral_point)
            self.N_ksi_matrix[i, 2] = ShapeFunctions.third_shape_function_dksi(integral_point)
            self.N_ksi_matrix[i, 3] = ShapeFunctions.fourth_shape_function_dksi(integral_point)
            i += 1

    def N_vector(ksi: float, eta: float):
        N_vector = np.zeros(shape=(4,1))
        integral_point = IntegralPoint(ksi, eta)
        N_vector[0, 0] = ShapeFunctions.first_shape_function(integral_point)
        N_vector[1, 0] = ShapeFunctions.second_shape_function(integral_point)
        N_vector[2, 0] = ShapeFunctions.third_shape_function(integral_point)
        N_vector[3, 0] = ShapeFunctions.fourth_shape_function(integral_point)

        return N_vector
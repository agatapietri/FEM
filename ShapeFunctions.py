from IntegralPoint import IntegralPoint


def first_shape_function(integral_point):
    return 0.25 * (1 - integral_point.ksi) * (1-integral_point.eta)


def second_shape_function(integral_point):
    return 0.25 * (1 + integral_point.ksi) * (1 - integral_point.eta)


def third_shape_function(integral_point):
    return 0.25 * (1 + integral_point.ksi) * (1 + integral_point.eta)


def fourth_shape_function(integral_point):
    return 0.25 * (1 - integral_point.ksi) * (1 + integral_point.eta)


def first_shape_function_dksi(integral_point):
    return -0.25 * (1 - integral_point.eta)


def second_shape_function_dksi(integral_point):
    return 0.25 * (1 - integral_point.eta)


def third_shape_function_dksi(integral_point):
    return 0.25 * (1 + integral_point.eta)


def fourth_shape_function_dksi(integral_point):
    return -0.25 * (1 + integral_point.eta)


def first_shape_function_deta(integral_point):
    return -0.25 * (1 - integral_point.ksi)


def second_shape_function_deta(integral_point):
    return -0.25 * (1 + integral_point.ksi)


def third_shape_function_deta(integral_point):
    return 0.25 * (1 + integral_point.ksi)

def fourth_shape_function_deta(integral_point):
    return 0.25 * (1 - integral_point.ksi)
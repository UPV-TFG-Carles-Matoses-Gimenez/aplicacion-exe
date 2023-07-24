from colour import LUT3D, LUT1D
import numpy as np
import matplotlib.pyplot as plt

def find_rectangular_matrix_dimensions(n):
    # Encuentra el valor entero más cercano a la raíz cuadrada de n
    closest_sqrt = round(np.sqrt(n))
    
    # Si el valor es un número entero, esa será la dimensión de la matriz cuadrada
    if closest_sqrt * closest_sqrt == n:
        return closest_sqrt, closest_sqrt
    
    # Si el valor no es un número entero, encuentra el número de filas y columnas apropiadas
    rows = np.ceil(np.sqrt(n))
    cols = np.ceil(n / rows)
    
    return int(cols), int(rows), 
    

def create_image_from_cube(cube):
    # Dims = XYZ, donde 
    # X = width
    # Y = height 
    # Z = puntos

    shape = cube.shape
    new_dims = find_rectangular_matrix_dimensions(shape[2])

    # Generamos imagen
    new_matrix = np.zeros((shape[0]*new_dims[0],shape[1]*new_dims[1],3))

    point = 0
    for i in range(new_dims[0]):
        for j in range(new_dims[1]):
            point+=1
            if point <= shape[2]:
                x_range= i*shape[0], i*shape[0]+shape[0]
                y_range= j*shape[1], j*shape[1]+shape[1]
                new_matrix[ x_range[0]:x_range[1] , y_range[0]:y_range[1] ] = cube[:,:,point-1,:]
    new_matrix = new_matrix[..., [1, 0, 2]]

    return new_matrix


def lut_generator(method, *args):
    args = args[0]

    LUT = LUT3D().linear_table(**args)
    image = create_image_from_cube(LUT)
    print(args)
    return LUT, image


LUT,image = lut_generator("",{"size":7})

plt.imshow(image);plt.show()
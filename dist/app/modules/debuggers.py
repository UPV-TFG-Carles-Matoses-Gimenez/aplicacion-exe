import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2 
import numpy as np

from modules import DEBUGGER_MENU, NODE_WINDOW_MENU
from modules.interaction import register

# from app import DEBUGGER_MENU

print("importing debuggers")
#####################################################
#####################################################

def array_max(*argss):
    # functions 
    
    return argss, np.max(argss)

class Array_max(NodeV2):
    Title = "array_max"
    def __init__(self):
        f = array_max
        inp_list = ["array"]
        out_list = ["array","max"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.custom_text = dpg.add_text("max: ",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        dpg.set_value(self.custom_text,f"max: {np.max(inp)}")
        return inp


#####################################################
#####################################################

def printer(val,self,*argss):
    dpg.set_value(self.custom_text,val)
    return val,

class Printer(NodeV2):
    Title = "Printer"
    def __init__(self):
        f = printer
        inp_list = ["val"]
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.custom_text = dpg.add_text("",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(self)
        return inp

#####################################################
#####################################################



def redirect(*argss):
    # Devuelve lista de mismos elementos que pins de salida
    return argss

# class Redirect(NodeV2):
#     Title = "redirect"
#     def __init__(self):
#         f = redirect
#         inp_list = ["val"]
#         out_list = ["val"]
#         super().__init__(f, inp_list, out_list, self.Title)
class Redirect(NodeV2):
    Title = "redirect"
    def __init__(self):
        super().__init__(redirect, ["val"], ["val"], self.Title)

#####################################################
#####################################################

class Window_debugger(NodeV2):
    Title = "Window_debugger"
    def __init__(self):
        super().__init__(redirect, ["val"], ["val"], self.Title)

        dpg.add_text("test_text", parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        print("no entiendo muy bien")
        return super().recollect_inputs_frombackawrd_nodes()


#####################################################
#####################################################

def array_generator(start,stop,num):
    return np.linspace(start,stop,num),

class Array_Generator(NodeV2):
    Title = "Array_Generator"
    def __init__(self):
        f = array_generator
        inp_list = {
            "start":{"float":{"width":120, "format": '%.8f', "default_value": 0}},
            "stop ":{"float":{"width":120,"format": '%.8f', "default_value": 10}},
            "num  ":{"integer":{"width":120, "default_value": 10}},
            }
        
        out_list = ["array"]
        super().__init__(f, inp_list, out_list, self.Title)

#####################################################
#####################################################

def image_generator(R_start,R_stop,G_start,G_stop,B_start,B_stop,num,pix_size):
    def convert_pixels_to_squares(image_array,cube_size):
        # Obtener las dimensiones de la imagen
        height, width, _ = image_array.shape
        # Crear una nueva matriz con el doble de altura y ancho
        new_height = height * cube_size
        new_width = width * cube_size
        new_image_array = np.zeros((new_height, new_width, 3), dtype=np.float64)
        
        # Iterar sobre cada píxel de la imagen original
        for i in range(height):
            for j in range(width):
                # Obtener el color del píxel
                color = image_array[i, j]
                
                # Calcular la posición del cuadrado en la nueva imagen
                square_x = j * cube_size
                square_y = i * cube_size
                
                # Pintar el cuadrado del mismo color en la nueva imagen
                new_image_array[square_y:square_y+cube_size, square_x:square_x+cube_size, :] = color
        
        return new_image_array

    image = np.linspace((R_start,G_start,B_start),(R_stop,G_stop,B_stop),num)
    _image = np.zeros((1,num,3))

    _image[0,:,0] = image[:,0]
    _image[0,:,1] = image[:,1]
    _image[0,:,2] = image[:,2]
    _image = convert_pixels_to_squares(_image,pix_size)
    return _image,

class Image_Generator(NodeV2):
    Title = "Image_Generator"
    def __init__(self):
        f = image_generator
        inp_list = {
            "R_start":{"float":{"width":120, "format": '%.8f', "default_value": 0}},
            "R_stop ":{"float":{"width":120,"format": '%.8f', "default_value": 10}},
            "G_start":{"float":{"width":120, "format": '%.8f', "default_value": 0}},
            "G_stop ":{"float":{"width":120,"format": '%.8f', "default_value": 10}},
            "B_start":{"float":{"width":120, "format": '%.8f', "default_value": 0}},
            "B_stop ":{"float":{"width":120,"format": '%.8f', "default_value": 10}},
            
            "num  ":{"integer":{"width":120, "default_value": 10}},
            "Pix_size  ":{"integer":{"width":120, "default_value": 10}},
            }
        
        out_list = ["array"]
        super().__init__(f, inp_list, out_list, self.Title)

#####################################################
#####################################################

####
####
# REGISTRO DE NODOS
with dpg.menu(label="DEBUGGER",tag=DEBUGGER_MENU,parent=NODE_WINDOW_MENU):
    pass

register_list = [
                 Printer, 
                 Array_max, 
                 Redirect, 
                 Array_Generator,
                 Image_Generator,

                 ]

for node in register_list:
    register(node, DEBUGGER_MENU)

####
####

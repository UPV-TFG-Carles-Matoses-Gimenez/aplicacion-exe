from modules.father_class import NodeV2
import dearpygui.dearpygui as dpg
import numpy as np

#####################################################
#####################################################
def exposition(val1,val2,*argss):
    return [val1**val2]

class Exposition(NodeV2):
    Title = "Exposition"
    def __init__(self):
        f = exposition
        inp_list = {"val":{"slider":{"width":120}},"exp":{"value":{"width":120,"format": '%.8f'}}}
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)

        dpg.add_text("Static example",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        # inp.append(dpg.get_value(self.type))
        return inp
    
    def custom_node_info(self):
        super().custom_node_info()
        dpg.add_text("Custom node info")
#####################################################
#####################################################

def adjust_saturation(percentage, image_array ):
    # gray image 
    gray_image = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = np.repeat(gray_image, 3, axis=-1)
    result = gray_image*(np.max([0,1-percentage])) + image_array*percentage

    return np.clip(result,0,1),

class Saturacion(NodeV2):
    Title = "Saturation"
    def __init__(self):
        f = adjust_saturation
        inp_list = {"sat":{"slider":{"width":120,"default_value":1,"min_value":0,"max_value":2}},"img":{"value":{"width":120,"format": '%.8f'}}}
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)


#####################################################
#####################################################


####
####
from modules.interaction import register
from modules import NODE_WINDOW_MENU
# REGISTRO DE NODOS
menu = "COLOR"
with dpg.menu(label=menu, tag=menu, parent=NODE_WINDOW_MENU):
    pass

register_list = [
    Exposition,
    Saturacion,

    ]
for node in register_list:
    register(node, menu)
####
####
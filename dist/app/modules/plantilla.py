from modules.father_class import NodeV2
import dearpygui.dearpygui as dpg

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

def temporal(img,*argss):
    return [val1**val2]

class Temporal(NodeV2):
    Title = "Temporal"
    def __init__(self):
        f = exposition
        inp_list = {"val":{"slider":{"width":120}},"img":{"value":{"width":120,"format": '%.8f'}}}
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


####
####
from modules.interaction import register
from modules import NODE_WINDOW_MENU
# REGISTRO DE NODOS
menu = "PLANTILLA"
with dpg.menu(label=menu, tag=menu, parent=NODE_WINDOW_MENU):
    pass

register_list = [
    Exposition,
    Temporal,
    ]
for node in register_list:
    register(node, menu)
####
####
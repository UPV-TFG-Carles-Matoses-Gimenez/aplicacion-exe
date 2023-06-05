import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2 
import numpy as np
#####################################################
#####################################################

def set_val(val,*argss):
    return val,

class Set_Val(NodeV2):
    Title = "Set Val"
    def __init__(self):
        f = set_val
        inp_list = []
        out_list = ["int"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.type = dpg.add_combo((
            "int",
            "float",
            ), 
            label="type",
            callback=lambda: self.custom_input(dpg.get_value(self.type),self.val,self.static,self.node_modified),
            parent=self.static,
            width=200,
            default_value="int")
        
        self.val = dpg.add_input_int(label="int",width=120,parent=self.static,callback=self.node_modified)

    def custom_input(self,str,val_id,parent_id,callback):
        dpg.delete_item(val_id)
        if str == "int":
            dpg.add_input_int(tag=val_id,label="int",width=120,parent=parent_id,callback=callback)
        elif str == "float":
            dpg.add_input_float(tag=val_id,label="float",width=120,parent=parent_id,callback=callback)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        

        inp.append(dpg.get_value(self.val))
        return inp

#####################################################
#####################################################

def divider(val1,val2,*argss):
    return val1/val2,

class Divider(NodeV2):
    Title = "Divider"
    def __init__(self):
        f = divider
        inp_list = ["val","val"]
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        return inp
    
#####################################################
#####################################################
def math_node_divide(val1,val2,*argss):
    return val1/val2
def math_node_muliply(val1,val2,*argss):
    return val1*val2

def math_node(val1,val2,type,*args):
    if type == "divide":
        val =  math_node_divide(val1, val2)
    elif type == "multiply":
        val =  math_node_muliply(val1, val2)
    else:
        val =  math_node_muliply(val1, val2)

    return np.array(val,dtype=np.float16),
    

class Math_Node(NodeV2):
    Title = "Math Node"
    def __init__(self):
        f = math_node
        inp_list = {"F1":{"slider":{"width":160}},"F2":{"value":{"width":160}}}
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.type = dpg.add_combo((
            "divide",
            "muliply",
            ), 
            label="",
            callback=lambda: self.node_modified,
            parent=self.static,
            width=150,
            default_value="multiply")

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.type))
        return inp
#####################################################
#####################################################

####
####
from modules import MATH, NODE_WINDOW_MENU
from modules.interaction import register
# REGISTRO DE NODOS
with dpg.menu(label=MATH, tag=MATH,parent=NODE_WINDOW_MENU):
    pass

register_list = [Math_Node,Set_Val,Divider]
for node in register_list:
    register(node, MATH)
####
####
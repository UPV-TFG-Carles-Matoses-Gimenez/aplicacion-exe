import dearpygui.dearpygui as dpg
from modules import INSPECTOR_WINDOW, interaction
import time
print("loading parent")
###################################################
###################################################

#### change node color theme
with dpg.theme() as theme_default:
    with dpg.theme_component(dpg.mvNode):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (40, 40, 40), category=dpg.mvThemeCat_Nodes)
        # dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

with dpg.theme() as theme_running:
    with dpg.theme_component(dpg.mvNode):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (40, 80, 40), category=dpg.mvThemeCat_Nodes)
        # dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

with dpg.theme() as theme_error:
    with dpg.theme_component(dpg.mvNode):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (100, 40, 40), category=dpg.mvThemeCat_Nodes)
        # dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

# FUNCTION HANDLERS
def node_selected(sender, app_data, user_data):
    print(f"sender {sender}")
    print(f"appdata {app_data}")
    print(f"user data {user_data}")
    interaction.inspector_callback()

with dpg.item_handler_registry(tag="Node Clicked") as handler:
    dpg.add_item_clicked_handler(callback=node_selected)

###################################################
###################################################

class NodeV2:
    super_list = []
    instances = []
    
    def __init__(self,f,inp_list:list,out_list:list,Title:str):
        self.__class__.instances.append(self)
        self.f = f
        self.nodeID = None
        self.static = None
        self.time_elapsed = None
        self.inp_list = inp_list
        self.out_list = out_list
        self.inputs = []
        self.outputs = []
        self.connections_forward = []  # {"input":app_data[1],"output":app_data[0],"instance":_i, "linkID":node_link}
        self.connections_backward = [] # {"input":app_data[1],"output":app_data[0],"instance":_o, "linkID":node_link}

        
        with dpg.node(label=Title, parent="node editor", user_data="test",) as self.nodeID:
            pass
        dpg.bind_item_theme(self.nodeID, theme_default)
        dpg.bind_item_handler_registry(self.nodeID, "Node Clicked")

        with dpg.node_attribute(parent=self.nodeID,attribute_type=dpg.mvNode_Attr_Static,user_data={}):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Debug",callback=self.debug)
                dpg.add_text(f"ID: {self.nodeID}")
                self.time_elapsed = dpg.add_text(f"TE: {0}")
        with dpg.node_attribute(parent=self.nodeID,attribute_type=dpg.mvNode_Attr_Static,user_data={}) as self.static:
            pass
            # with dpg.child_window(autosize_x=True,autosize_y=True) as self.static:
            
        
        # Construct inputs
        self.add_pins_inp(inp_list)
        self.add_pins_out(out_list)
        self.node_info()


    def debug(self,*args):
        print("\n")
        print(f"Node: {self.nodeID}")
        print(f"connections_forward {self.connections_forward}")
        print(f"connections_backward {self.connections_backward}")
        print("connections to end: ",self.get_all_forward_nodes())
        print("superlist: ",self.super_list)
        
        for i in self.outputs:
            a = dpg.get_item_user_data(i)
            print(a)

        print("\n")
        return [
            "\n",
            f"Node: {self.nodeID}",
            f"connections_forward {self.connections_forward}",
            f"connections_backward {self.connections_backward}",
            f"connections to end: {self.get_all_forward_nodes()}",
            f"superlist: {self.super_list}"
            ]

    def _toggle_item_visibility(self, ID):
        if ID == None:
            return
        
        visibility = dpg.is_item_visible(ID)
        print(f"toggle ID VISIBILITY: {visibility}")
        if visibility:
            dpg.hide_item(ID)
        else:
            dpg.show_item(ID)

    def get_default_pin_ID(self, pinID):
        groupID = dpg.get_item_children(pinID,1)[0]
        special_inputID=dpg.get_item_children(groupID,1)

        if len(special_inputID) > 1:
            return special_inputID[1]
        return

    def get_default_pin_val(self, pinID):
        groupID = dpg.get_item_children(pinID,1)[0]
        special_inputID=dpg.get_item_children(groupID,1)

        if len(special_inputID) > 1:
            return dpg.get_value(special_inputID[1])
        return

    def add_pins_inp(self,pins_list):

        if isinstance(pins_list, list):
            my_dict = dict.fromkeys(pins_list, None)
        else:
            my_dict = pins_list

        for i in my_dict:
            
            # SI PIN NORMAL
            if my_dict[i] == None:
                with dpg.node_attribute(parent=self.nodeID,attribute_type=dpg.mvNode_Attr_Input) as inp:
                    with dpg.group(horizontal=True): 
                        dpg.add_text(f"{i}") ####

            # SI PIN ESPECIAL
            else:
                with dpg.node_attribute(parent=self.nodeID,attribute_type=dpg.mvNode_Attr_Input) as inp:
                    # {"F1":{"slider":{"height":60, "width":20}},"F2":{"slider":{"height":60,"width":60}}}
                    _label = i                                        # F1
                    _inp_dict = my_dict[_label]                       # {"slider":{"height":60, "width":20}}
                    _inp_dict_dict_key = list(_inp_dict.keys())[0]    # "slider"
                    _inp_dict_dict_val =_inp_dict[_inp_dict_dict_key] # {"height":60, "width":20}

                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{i}") ####
                        match _inp_dict_dict_key:
                            case "slider":
                                dpg.add_slider_double(callback=self.node_modified,**_inp_dict_dict_val)
                            case "color":
                                dpg.add_text(label="not implemented",callback=self.node_modified,**_inp_dict_dict_val)
                            case "value":
                                dpg.add_input_double(callback=self.node_modified,**_inp_dict_dict_val)
                            case "float":
                                dpg.add_input_double(callback=self.node_modified,**_inp_dict_dict_val)
                            case "integer":
                                dpg.add_input_int(callback=self.node_modified,**_inp_dict_dict_val)
                            case "text":
                                dpg.add_input_text(callback=self.node_modified,**_inp_dict_dict_val)
                            case "none":
                                dpg.add_text()
                            case "button":
                                dpg.add_button(**_inp_dict_dict_val)
                                
            self.inputs.append(inp)


    def add_pins_out(self,pins_list):
        for i in pins_list:
            with dpg.node_attribute(parent=self.nodeID,attribute_type=dpg.mvNode_Attr_Output) as out:
                self.outputs.append(out)
                dpg.add_text(f"{i}")

    def remove_pins_out(self):
        for i in self.outputs:
            dpg.delete_item(i)
        self.outputs = []

    def set_outputs(self,*args):
        print("argumentos para los outputs")
        print(len(args))
        for i in range(len(self.outputs)):
            dpg.set_item_user_data(self.outputs[i],args[i])
            print(f"set output: {self.outputs[i]} to {type(args[i])}")

    def get_all_forward_nodes(self):
        forward_nodes_conected_list = [] # Parent IDs

        # This Node
        for i in self.connections_forward:
            # {"input":app_data[1],"output":app_data[0],"instance":_i, "linkID":node_link}
            parent_id = dpg.get_item_parent(i["input"])
            if parent_id not in forward_nodes_conected_list:
                forward_nodes_conected_list.append(parent_id)

        # Next Nodes
        longitud = len(forward_nodes_conected_list)
        new_longitud = len(forward_nodes_conected_list)+1

        while longitud < new_longitud: # mientras no se añada un nuevo elemento
            longitud  = len(forward_nodes_conected_list)
            for i in forward_nodes_conected_list: # futuros nodos (formato ID)
                for node in self.instances: # obtener la instancia
                    if node.nodeID == i: # obtener la instancia
                        for i in node.connections_forward:
                            parent_id = dpg.get_item_parent(i["input"])
                            if parent_id not in forward_nodes_conected_list:
                                forward_nodes_conected_list.append(parent_id)
            new_longitud = len(forward_nodes_conected_list)
        return forward_nodes_conected_list
    
    def remove_from_superlist(self):
        # Determinar los outputs de los nodos requeridos
        # substraer IDs de superlist 
        forward_list = self.get_all_forward_nodes()
        for i in forward_list:
            if i in self.super_list:
                self.super_list.remove(i)

    def get_class_reference(self,id):
        for node in self.instances: # obtener la instancia
            if node.nodeID == id: # obtener la instancia
                return node

    def node_modified(self):
        
        #### Borrar todos los nodos en la superlista que requieran refrescarse 
        self.remove_from_superlist() # elimina todos los nodos que requieren renderizarse
        self.refresh() # actualiza los propios outputs
        print("Refreshing future nodes")
        self.refresh_forward_nodes() # actualizar nodos de la superlista

    def refresh_forward_nodes(self):
        for i in self.get_all_forward_nodes():
            node_id = self.get_class_reference(i)
            if node_id not in self.super_list:
                node_id.refresh()

    def recollect_inputs_frombackawrd_nodes(self):
        # GESTION DE INPUT O VALOR DEFAULT
        orden = {}
        lista = [None]*len(self.inputs)

        # Siempre default 
        for iter,input in enumerate(self.inputs):
            lista[iter] = self.get_default_pin_val(input)
            print(input)

        # Excepto si conexion
        for id in range(len(self.inputs)):
            for con in self.connections_backward:
                if self.inputs[id] == con["input"]:
                    data = con.copy()
                    lista[id] = dpg.get_item_user_data(data["output"])
        print(f"inputs {self.inputs} to {lista}")
        return lista
    
    def stetic_change(self,*args):
        pass

    def check_out(self,out):
        if out is None:
            return False
        elif isinstance(out, list):
            return True
        else:
            return True

    def refresh(self):
        """
        1. comprobar que tiene todos los inputs necesarios
        2. actualizar outputs
        """
        print("\n")
        print(f"starting function: {self.nodeID}")
        # comprobar que todos los nodos anteriores estan renderizados 
        # {"input":app_data[1],"output":app_data[0],"instance":_o, "linkID":node_link}
        for i in self.connections_backward:
            node_id = i["instance"].nodeID #id del padre
            if node_id not in self.super_list:
                # requiere refrescar
                # buscar la referencia a la clase
                ref = self.get_class_reference(node_id)
                ref.refresh()
                if ref.nodeID not in ref.__class__.super_list:
                    ref.__class__.super_list.append(ref.nodeID)
                
        ### Comprobar inputs
        # 1. Recolectar inputs
        inp = self.recollect_inputs_frombackawrd_nodes() # Lista con los inputs
        print(f"function: {self.nodeID} with input: {type(inp)}")

        try:
            # 2. ejecutar funcion
            start_time = time.time()
            dpg.bind_item_theme(self.nodeID, theme_running)

            if inp: # si hay inputs en el nodo
                print("Todo correcto")
                out = self.f(*inp)
            else: # si no tiene inputs el nodo
                out = self.f()

            end_time = time.time()
            elapsed_time = end_time - start_time  
            dpg.set_value(self.time_elapsed,f"TE: {elapsed_time:.2f}")
            dpg.bind_item_theme(self.nodeID, theme_default)

            print(f"Funcion finalizada: {type(out)}")

            # set outputs
            self.set_outputs(*out)

            # Se añade a la superlista
            if self.nodeID not in self.__class__.super_list:
                self.__class__.super_list.append(self.nodeID)
            
            print(f"Succes!! refresh: {self.nodeID}")
            print("\n")
            return out
        
        except Exception as e:
            print("Se ha producido una excepción:", type(e).__name__)
            print(e)
            for i in self.outputs:
                dpg.set_item_user_data(i,None) #Todos los outputs a None
            dpg.bind_item_theme(self.nodeID, theme_error)
            return
        
    
    def _delete(self):
        # instancia de nodos anteriores y posteriores 
        for B in self.connections_backward:
            P = B["instance"]
            eliminar = {
                "input":B["input"],
                "output":B["output"],
                "instance":self,
                "linkID":B["linkID"]
                }
            if eliminar in P.connections_forward:
                print(f"el diccionario {eliminar}\nSe elimina de {P.connections_forward}")
                P.connections_forward.remove(eliminar)

        for F in self.connections_forward:
            P = F["instance"]
            eliminar = {
                "input":F["input"],
                "output":F["output"],
                "instance":self,
                "linkID":F["linkID"]
                }
            if eliminar in P.connections_backward:
                print(f"el diccionario {eliminar}\nSe elimina de {P.connections_backward}")
                P.connections_backward.remove(eliminar)

            
            # Set actual conections to None
        childrens_list = dpg.get_item_children(self.nodeID) 
        print(f"childrens_list {childrens_list}")
        if 1 in childrens_list.keys():
            print("1 is in keys")
            for i in childrens_list[1]:
                print(f"i to None {i}")
                dpg.set_item_user_data(i,{})

            # Refresh nodes conected
        # self.refresh_connections_forward()

            # delete all conections
        self.remove_all_connections_forward() # delete item link and empty connections
        self.remove_all_connections_backward() # delete item link and empty connections
 
        dpg.delete_item(self.nodeID, children_only=True) # delete pins

        # Delete Inspector
        dpg.delete_item(self.window_info)

        self.instances.remove(self)
        dpg.delete_item(self.nodeID)

    def remove_all_connections_forward(self):
        for i in self.connections_forward:
            dpg.delete_item(i["linkID"]) 
        self.connections_forward = [] # Borrar todas las conexiones

    def remove_all_connections_backward(self):
        for i in self.connections_backward:
            dpg.delete_item(i["linkID"]) 
        self.connections_backward = [] # Borrar todas las conexiones

    def split_string(self, long_string, chunk_size):
        chunks = [long_string[i:i+chunk_size] for i in range(0, len(long_string), chunk_size)]
        return "\n".join(chunks)

    def node_info(self):
        def close_info(sender, app_data, user_data):
            dpg.hide_item(user_data)

        cabecera = f"{self.Title} = {self.nodeID}"
        with dpg.child_window(label=cabecera,parent=INSPECTOR_WINDOW, menubar=True, height=400, show=False) as self.window_info: #, autosize_y=True
            # menu bar
            with dpg.menu_bar(label="father menu"):
                dpg.add_menu_item(label="X", user_data= self.window_info, callback=close_info)
            
            # debugg info
            dpg.add_text(cabecera)
            with dpg.tree_node(label="Debug", default_open=False):
                for i in self.debug():
                    dpg.add_text(self.split_string(i,50))
                            
            self.custom_node_info()
            # Custom info ...
       
    def custom_node_info(self):
        pass


print("Parent loaded!!")
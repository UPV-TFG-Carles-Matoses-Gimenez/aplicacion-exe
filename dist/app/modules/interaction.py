"""Relation between nodes"""
import time
import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2
from modules import INSPECTOR_WINDOW, NODE_WINDOW, NODE_WINDOW_EDITOR


def register(node, parent):
    print(f"REGISTERING NODE {node.Title}")
    dpg.add_menu_item(parent=parent, label=node.Title, callback=lambda:node())

def check_if_pin_in_use(output, input):
    # a que nodo vamos a conectarnos?
    instancia_a_conectar_a = get_unic_class(input)
    out_pins_conectados = instancia_a_conectar_a.connections_backward
    for i in out_pins_conectados:
        if i["input"] == input: # ya esta en uso

            default_pin_ID = instancia_a_conectar_a.get_default_pin_ID(input)
            instancia_a_conectar_a._toggle_item_visibility(default_pin_ID)

            print(f"conexión {output}:{input} ya existe")
            instancia_a_conectar_a.connections_backward.remove(i)

                # en el nodo anterior al que queremos conectarnos
            dict_ = {"input":input,"output":i["output"], "instance":instancia_a_conectar_a, "linkID":i["linkID"]}
            i["instance"].connections_forward.remove(dict_)
            print(dict_)
                # en objeto link
            dpg.delete_item(i["linkID"]) 



def get_unic_class(id):
    """From pin to parent"""
    for instance in NodeV2.instances:
        if instance.nodeID == dpg.get_item_parent(id):
            return instance
    return

def get_class_from_id(id):
    for instance in NodeV2.instances:
        if instance.nodeID == id:
            return instance
    return

def get_class(output, input):
    # ACTUALIZA el nodo input conectado
    # la clase que contiene el nodo posee la misma ID que el nodo
    print("get node: ",dpg.get_item_parent(input))
    for instance in NodeV2.instances:
        if instance.nodeID == dpg.get_item_parent(input):
            _i = instance
        if instance.nodeID == dpg.get_item_parent(output):
            _o = instance
    return _i, _o

def link_callback(sender, app_data, user_data):
    check_if_pin_in_use(app_data[0],app_data[1])
    
    # app_data -> (link_id1, link_id2)
    node_link = dpg.add_node_link(app_data[0], app_data[1], parent=sender,label="link")
    dpg.set_item_user_data(node_link, {"input":app_data[1],"output":app_data[0], "linkID":node_link})
    print("Connection: ", {"input":app_data[1],"output":app_data[0], "linkID":node_link})
    
    # Nodo input y output
    _i,_o = get_class(app_data[0],app_data[1])
    #  # guarda la conexión dentro de la clase (se guarda en el nodo OUTPUT)
    _i.connections_backward.append({"input":app_data[1],"output":app_data[0],"instance":_o, "linkID":node_link})
    #  # guarda la conexión dentro de la clase (se guarda en el nodo INPUT)
    _o.connections_forward.append({"input":app_data[1],"output":app_data[0],"instance":_i, "linkID":node_link})
    # Almacenar output
    _i.node_modified()  #Refrescar nodo actual, Le da valor a los outputs de el mismo

    # Gestion estetica del pin
    default_pin_ID = _i.get_default_pin_ID(app_data[1])
    _i._toggle_item_visibility(default_pin_ID)


# callback runs when user attempts to disconnect attributes
def delink_callback(sender, app_data):
    # app_data -> link_id with data:
    # {"input":app_data[1],"output":app_data[0], "linkID":node_link}
    i_o = dpg.get_item_user_data(app_data)
    # reset input node
    _i,_o = get_class(i_o["output"],i_o["input"]) #pins to class father
    print(_o.connections_forward)
    _o.connections_forward.remove ({"input":i_o["input"],"output":i_o["output"],"instance":_i, "linkID":app_data})
    _i.connections_backward.remove({"input":i_o["input"],"output":i_o["output"],"instance":_o, "linkID":app_data})
    dpg.delete_item(app_data)
    print("link {}, input {}, output {} REMOVED".format(app_data,i_o["input"],i_o["output"]))
    _i.node_modified()

    # Gestion estetica del pin
    default_pin_ID = _i.get_default_pin_ID(i_o["input"])
    _i._toggle_item_visibility(default_pin_ID)


def split_string(long_string, chunk_size):
    chunks = [long_string[i:i+chunk_size] for i in range(0, len(long_string), chunk_size)]
    return "\n".join(chunks)

def inspector_callback():
    for i in NodeV2.instances:
        dpg.hide_item(i.window_info)

    cl_list =  dpg.get_selected_nodes(NODE_WINDOW_EDITOR)
    for cls in cl_list:
        cls = get_class_from_id(cls)
        dpg.show_item(cls.window_info)


# Del node on ctrl X
def on_key_la(sender, app_data):

    if dpg.is_key_down(dpg.mvKey_X):
        print("Ctrl + X")
        node_list = dpg.get_selected_nodes(NODE_WINDOW_EDITOR)
        print(node_list)
        for i in node_list:
            for instance in NodeV2.instances:
                if instance.nodeID ==i:
                    print(f"removeing {i}")
                    instance._delete()

# Duplicate node
def on_key_la2(sender, app_data):

    if dpg.is_key_down(dpg.mvKey_D):
        print("Ctrl + D")
        node_list = dpg.get_selected_nodes(NODE_WINDOW_EDITOR)
        for i in node_list:
            for instance in NodeV2.instances:
                if instance.nodeID ==i:
                    print(f"duplicating {i}")
                    instance.__class__().__init__


# Del node on ctrl X
def on_key_X(sender, app_data):
        print("X")
        node_list = dpg.get_selected_nodes(NODE_WINDOW_EDITOR)
        print(node_list)
        for i in node_list:
            for instance in NodeV2.instances:
                if instance.nodeID ==i:
                    print(f"removeing {i}")
                    instance._delete()

# Duplicate node
def on_key_D(sender, app_data):
        print("D")
        node_list = dpg.get_selected_nodes(NODE_WINDOW_EDITOR)
        for i in node_list:
            for instance in NodeV2.instances:
                if instance.nodeID ==i:
                    print(f"duplicating {i}")
                    instance.__class__().__init__


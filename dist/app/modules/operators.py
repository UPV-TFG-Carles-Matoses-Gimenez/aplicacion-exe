import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2
import numpy as np
import cv2
# import colour

#####################################################
#####################################################

def multiply(val1,val2,*argss):
    return val1*val2,

class Multiply(NodeV2):
    Title = "Multiply"
    def __init__(self):
        f = multiply
        inp_list = ["val1","val2"]
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.custom_text = dpg.add_text("modificar",parent=self.static)

    def refresh(self):
        out = super().refresh()
        if self.check_out(out):
            dpg.set_value(self.custom_text,out)

#####################################################
#####################################################

def white_balance_raw(matrix,matrix_bayer,wb,*argss):
    # functions 
    red = np.where(matrix_bayer==0,True,False)
    green1 = np.where(matrix_bayer==1,True,False)
    green2 = np.where(matrix_bayer==3,True,False)
    blue = np.where(matrix_bayer==2,True,False)

    matrix[red]    = matrix[red]      *wb[0]
    matrix[green1] = matrix[green1]   *wb[1]
    matrix[blue]   = matrix[blue]     *wb[2]
    matrix[green2] = matrix[green2]   *wb[3]
    
    return matrix,

class White_Balance_Raw(NodeV2):
    Title = "White Balance"
    def __init__(self):
        f = white_balance_raw
        inp_list = ["matrix","matrix bayer","wb"]
        out_list = ["matrix"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.custom_text = dpg.add_text("white balance: ",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        wb = inp[2]
        dpg.set_value(self.custom_text,wb)
        return inp

#####################################################
#####################################################

# import colour
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from colour_demosaicing import demosaicing_CFA_Bayer_DDFAPD
from colour_demosaicing import mosaicing_CFA_Bayer
from colour_demosaicing import masks_CFA_Bayer

def demosaico(matrix,pattern,mode,*args):
    # functions 
    print("procesando demosaico")
    if mode == "demosaicing_CFA_Bayer_bilinear":
        m=demosaicing_CFA_Bayer_bilinear(matrix,pattern)
    elif mode == "demosaicing_CFA_Bayer_Malvar2004":
        m=demosaicing_CFA_Bayer_Malvar2004(matrix,pattern)
    elif mode == "demosaicing_CFA_Bayer_Menon2007":
        m=demosaicing_CFA_Bayer_Menon2007(matrix,pattern)
    elif mode == "demosaicing_CFA_Bayer_DDFAPD":
        m=demosaicing_CFA_Bayer_DDFAPD(matrix,pattern)
    elif mode == "mosaicing_CFA_Bayer":
        m=mosaicing_CFA_Bayer(matrix,pattern)
    elif mode == "masks_CFA_Bayer":
        m=masks_CFA_Bayer(matrix,pattern)

    return m,

class Demosaico(NodeV2):
    Title = "Demosaico"
    def __init__(self):
        f = demosaico
        inp_list = ["matrix"]
        out_list = ["matrix"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.demosaic_mode = dpg.add_combo((
                        "demosaicing_CFA_Bayer_bilinear",
                        "demosaicing_CFA_Bayer_Malvar2004", 
                        "demosaicing_CFA_Bayer_Menon2007",
                        "demosaicing_CFA_Bayer_DDFAPD",
                        "mosaicing_CFA_Bayer",
                        "masks_CFA_Bayer",
                        ), 
                      label="Demosaico",callback=self.node_modified,parent=self.static,width=200,default_value="demosaicing_CFA_Bayer_bilinear")
        self.pattern_mode = dpg.add_combo((
                        "RGGB",
                        "BGGR",
                        "GRBG",
                        "GBRG",
                        ), 
                      label="Pattern",callback=self.node_modified,parent=self.static,width=200,default_value="RGGB")


    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.pattern_mode))
        inp.append(dpg.get_value(self.demosaic_mode))
        print("input demosaico ",inp)
        return inp

#####################################################
#####################################################

def resize(img,scale_percent,*argss):
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Resize the image to the new dimensions
    img = np.array(img,dtype=np.float32)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR )
    return np.array(resized_img,dtype=np.float16),

class Resize(NodeV2):
    Title = "Resize"
    def __init__(self):
        f = resize
        inp_list = ["img"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.slider = dpg.add_slider_int(width=200,label="scale",min_value=1,max_value=100,default_value=50,callback=self.node_modified,parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.slider))
        return inp
    
#####################################################
#####################################################


def basic_hyperbola(x,s_y,s_x,p,*argss):
    x = np.array(x,dtype=np.float16)
    return np.array((s_y * (x / (x + s_x))*p),dtype=np.float16),

class Gamma(NodeV2):
    Title = "Gamma"
    def __init__(self):
        f = basic_hyperbola
        inp_list = ["img"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.s_y = dpg.add_slider_double(label="y",width=150,min_value=1,max_value=1.1, default_value=1,callback=self.node_modified,parent=self.static)
        self.s_x = dpg.add_slider_double(label="x",width=150,min_value=0.05,max_value=4,default_value=1,callback=self.node_modified,parent=self.static)
        self.p   = dpg.add_slider_double(label="p",width=150,min_value=1,max_value=1.6, default_value=1,callback=self.node_modified,parent=self.static)


        with dpg.plot(label="Hyperbola", 
                        height=150,
                        width=150,
                        no_title=True,
                        equal_aspects=True,
                        parent=self.static,

            ) as  self.plot:
            # REQUIRED: create x and y axes
            dpg.              add_plot_axis(dpg.mvXAxis,no_tick_marks=True,no_tick_labels=True)
            self.series = dpg.add_plot_axis(dpg.mvYAxis,no_tick_marks=True,no_tick_labels=True)
            self.line_series = dpg.add_line_series(np.linspace(0,1,1000),np.linspace(0,1,1000),parent=self.series)



    def recollect_inputs_frombackawrd_nodes(self):
        s_y = dpg.get_value(self.s_y) # overall clip
        s_x = dpg.get_value(self.s_x)# exposure
        p   = dpg.get_value(self.p) # shadow compresion
        curve = basic_hyperbola(np.linspace(0,1,1000),s_y,s_x,p)[0]
        dpg.set_value(self.line_series, [np.linspace(0,1,1000), curve.astype(np.float32)])

        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(s_y)
        inp.append(s_x)
        inp.append(p)
        return inp
    
#####################################################
#####################################################

def clip(img,min,max,*argss):
    if max == 0:
        max=None

    img = np.clip(img,min,max)
    return img,

class Clip(NodeV2):
    Title = "Clip"
    def __init__(self):
        f = clip
        inp_list = ["val"]
        out_list = ["val"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.min = dpg.add_input_float(label="min",default_value=0,width=120,parent=self.static,callback=self.node_modified)
        self.max = dpg.add_input_float(label="max",default_value=1,width=120,parent=self.static,callback=self.node_modified)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.min))
        inp.append(dpg.get_value(self.max))
        return inp

#####################################################
#####################################################

def regenerate_channel(bayer_image, matrix_bayer,lim,*args):

    x,y = np.where((matrix_bayer==1)) 
    for i in zip(x,y):
        if bayer_image[i] >= lim:
            p1,p2 = i
            bayer_image[p1,p2] = (bayer_image[p1,p2-1]+bayer_image[p1-1,p2])/2
            bayer_image[p1+1,p2-1] = (bayer_image[p1,p2-1]+bayer_image[p1-1,p2])/2
    return bayer_image,

class Regenerate_Channel(NodeV2):
    Title = "Regenerate_Channel"
    def __init__(self):
        f = regenerate_channel
        inp_list = ["raw matrix", "matrix_bayer", "lim"]
        out_list = ["raw matrix"]
        super().__init__(f, inp_list, out_list, self.Title)


#####################################################
#####################################################

def remove_alpha(img,*args):
    return img[:,:,0:3],

class Remove_Alpha(NodeV2):
    Title = "Remove_Alpha"
    def __init__(self):
        f = remove_alpha
        inp_list = ["img"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)


#####################################################
#####################################################
def float_to_rgbe(image, *, channel_axis=-1):

    # ensure channel-last
    image = np.moveaxis(image, channel_axis, -1)

    max_float = np.max(image, axis=-1)
    
    scale, exponent = np.frexp(max_float)
    scale *= 256.0/max_float

    image_rgbe = np.empty((*image.shape[:-1], 4))
    image_rgbe[..., :3] = image * scale
    image_rgbe[..., -1] = exponent + 128

    image_rgbe[scale < 1e-32, :] = 0
    
    # restore original axis order
    image_rgbe = np.moveaxis(image_rgbe, -1, channel_axis)

    return image_rgbe

def split_channels(img,self,*args):
    pin_list = []
    channels = []

    if dpg.get_value(self.rgbe):
        img = float_to_rgbe(img)
        
    n_channels = img.shape[2] if img.ndim >= 3 else 1
    print("Check n_channels ",n_channels)

    if self.n_canales_anterior == n_channels:
        print("mismas dimensiones, no borrar outputs")
        if n_channels == 1:
            # self.add_pins_out(["Gray"])
            return img,
        
        for i in range(n_channels):
            # pin_list.append(str(i))
            channels.append(img[:,:,i])

        # self.add_pins_out(pin_list)
        return channels
    
    else: 
        print("Borrar los outputs por los nuevos")
        self.remove_all_connections_forward()
        self.remove_pins_out()
        self.n_canales_anterior = n_channels
        if n_channels == 1:
            self.add_pins_out(["Gray"])
            return img,
        
        for i in range(n_channels):
            pin_list.append(str(i))
            channels.append(img[:,:,i])

        self.add_pins_out(pin_list)
        
        return channels


class Split_Channel(NodeV2):
    Title = "Split_Channel"
    def __init__(self):
        f = split_channels
        inp_list = ["img"]
        out_list = []
        super().__init__(f, inp_list, out_list, self.Title)

        self.rgbe = dpg.add_checkbox(label="RGBE", default_value=False,parent=self.static)
        self.n_canales_anterior = 0

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(self)
        return inp
#####################################################
#####################################################



####
####
from modules import OPERATORS, NODE_WINDOW_MENU
from modules.interaction import register
# REGISTRO DE NODOS
with dpg.menu(label=OPERATORS, tag=OPERATORS,parent=NODE_WINDOW_MENU):
    pass

register_list = [
    Resize,
    White_Balance_Raw,
    Demosaico,
    Gamma,
    Clip,
    Regenerate_Channel,
    Remove_Alpha,
    Split_Channel,
    ]
for node in register_list:
    register(node, OPERATORS)
####
####
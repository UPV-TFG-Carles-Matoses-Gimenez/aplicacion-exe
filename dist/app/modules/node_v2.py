import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2
import numpy as np


#####################################################

def int_test(in1,*args):
    in1*=2
    return in1,
class t1(NodeV2):
    Title = "Multiply *2"
    def __init__(self):
        f = int_test
        inp_list = ["int"]
        out_list = ["int"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.custom_text = dpg.add_text("valor: ",parent=self.static)

    def refresh(self):
        out = super().refresh()
        if self.check_out(out):
            dpg.set_value(self.custom_text,f"{out}")

#####################################################
#####################################################

def random(*args):
    return np.random.randint(0,10,4)
class t2(NodeV2):
    Title = "Random Generator"
    def __init__(self):
        f = random
        inp_list = []
        out_list = ["int","int","int","int"]
        super().__init__(f, inp_list, out_list, self.Title)

        dpg.add_button(label="generate", parent=self.static, callback=self.node_modified)
        self.custom_text = dpg.add_text("modificar",parent=self.static)

    def refresh(self):
        out = super().refresh()
        if self.check_out(out):
            dpg.set_value(self.custom_text,out)

#####################################################
#####################################################

def multiply(val1,val2,*argss):
    return val1*val2,

class t3(NodeV2):
    Title = "multiply"
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

from PIL import Image
def load_image(path,*args):
    """get:
    channels [R,G,B]
    """
    print("load image con path: ",path)
    if path != None:
        # PROCESO DE IMAGEN
        img = Image.open(path) 
        img = np.array(img)
        if img is None:
            print("OPEN CV NO PUEDE LEER LA IMAGEN")
            return None,

        # normalizar imagenes
        if img.dtype == np.uint8:
            img = np.array(img,dtype=np.float16)
            img /= 255
    
        # ENDING REFRESH FUNCTIONS
        print("image_refresh, set loadder output")
        return img,

class Load_Image(NodeV2):
    Title = "Load image"
    path = None
    def __init__(self):
        f = load_image
        inp_list = []
        out_list = ["channels"]
        super().__init__(f, inp_list, out_list, self.Title)
        
        # FILE DIALOG 
        def load_image_callback(sender, app_data, user_data):
            print('OK was clicked.')
            print("Sender: ", sender)
            get_dictionary = app_data['selections']
            for path in get_dictionary.values(): # extract path
                pass
            self.path=path
            print("app_data path: ", path)

            self.node_modified()

        # search item 
        with dpg.file_dialog(
            min_size=[100,200],
            directory_selector=False,
            show=False,
            callback=load_image_callback, 
            # cancel_callback=load_image_cancel_callback
            ) as file_dialog:

            dpg.add_file_extension(".*")
            dpg.add_file_extension(".png", 
                                color=(0, 255, 0, 255), 
                                custom_text="[png]")
        # FILE DIALOG #############

        dpg.add_button(parent=self.static,
                       label=" \n load image \n ",
                       callback=lambda:dpg.show_item(file_dialog)
                       )
        
        self.custom_text = dpg.add_text("img...",parent=self.static)

    def refresh(self):
        out = super().refresh()
        if self.check_out(out):
            dpg.set_value(self.custom_text,self.split_string(self.path,25))

    def recollect_inputs_frombackawrd_nodes(self):
        return self.path,

#####################################################
#####################################################
import rawpy

def load_image_raw(path,*args):
    print("child refresh")
    # PROCESO DE IMAGEN
    if path == None:
        return

    raw = rawpy.imread(path)
    matrix_bayer   = raw.raw_colors
    matrix         = raw.raw_image
    pt             = raw.raw_pattern
    black          = raw.black_level_per_channel
    postproced     = raw.postprocess()
    wb             = raw.camera_whitebalance
    rgb_xyz_matrix = raw.rgb_xyz_matrix[:3]
    white_level    = raw.white_level
    sizes          = raw.sizes
        
    # 1 eliminar negro ((falta mejorarlo))
    matrix = matrix - 0
    matrix_blacklevel = matrix-black[0]

    return matrix_blacklevel, matrix_bayer, wb, rgb_xyz_matrix, white_level, pt

class Load_Image_Raw(NodeV2):
    Title = "Load image Raw"
    path = None
    def __init__(self):
        f = load_image_raw
        inp_list = []
        out_list = ["matrix_blacklevel","matrix_bayer","wb","rgb_xyz_matrix","white_level","pattern"]
        super().__init__(f, inp_list, out_list, self.Title)
        
        # FILE DIALOG 
        def load_image_callback(sender, app_data, user_data):
            print('OK was clicked.')
            print("Sender: ", sender)
            get_dictionary = app_data['selections']
            for path in get_dictionary.values(): # extract path
                pass
            self.path=path
            print("app_data path: ", path)

            self.node_modified()

        # search item 
        with dpg.file_dialog(
            min_size=[100,200],
            directory_selector=False,
            show=False,
            callback=load_image_callback, 
            ) as file_dialog:

            dpg.add_file_extension(".*")
            dpg.add_file_extension(".NEF", color=(0, 255, 0, 255), custom_text="[NEF]")
            dpg.add_file_extension(".nef", color=(0, 255, 0, 255), custom_text="[nef]")
            dpg.add_file_extension(".CR2", color=(0, 255, 0, 255), custom_text="[CR2]")
            dpg.add_file_extension(".SR2", color=(0, 255, 0, 255), custom_text="[SR2]")
            dpg.add_file_extension(".PTX", color=(0, 255, 0, 255), custom_text="[PTX]")
            dpg.add_file_extension(".ORF", color=(0, 255, 0, 255), custom_text="[ORF]")
            dpg.add_file_extension(".RAF", color=(0, 255, 0, 255), custom_text="[RAF]")
            dpg.add_file_extension(".RAW2", color=(0, 255, 0, 255), custom_text="[RAW2]")
        # FILE DIALOG #############
        # FILE DIALOG #############

        dpg.add_button(parent=self.static,
                       label=" \n load image \n ",
                       callback=lambda:dpg.show_item(file_dialog)
                       )
        
        self.custom_text = dpg.add_text("img...",parent=self.static)

    def refresh(self):
        out = super().refresh()
        if self.check_out(out):
            dpg.set_value(self.custom_text,self.split_string(f"img: {self.path}",25))

    def recollect_inputs_frombackawrd_nodes(self):
        return self.path,

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

def printer(*argss):
    # functions 
    print(argss)
    return argss,

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
        dpg.set_value(self.custom_text,inp)
        return inp

#####################################################
#####################################################

import cv2
import colour
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

def rgb_to_xyz(matrix,rgb_xyz_matrix,*argss):
    RGB_to_XYZ_matrix = np.linalg.inv(rgb_xyz_matrix) #Matriz del archivo raw M^-1 a M
    
    # futura implementacio
    arreglo = [0.486493, 1.08008, 0.8887] # multiplicar RGB por coeficientes para conseguir D65 en la matriz M
    RGB_to_XYZ_matrix[:,0]=RGB_to_XYZ_matrix[:,0]*arreglo[0]
    RGB_to_XYZ_matrix[:,1]=RGB_to_XYZ_matrix[:,1]*arreglo[1]
    RGB_to_XYZ_matrix[:,2]=RGB_to_XYZ_matrix[:,2]*arreglo[2]

    new_channels = colour.RGB_to_XYZ(
        matrix,
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']["D65"],
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']["D65"],
        RGB_to_XYZ_matrix,
        chromatic_adaptation_transform = "Bradford"
            )
    return new_channels,

class RGB_to_XYZ(NodeV2):
    Title = "RGB_to_XYZ"
    def __init__(self):
        f = rgb_to_xyz
        inp_list = ["matrix","rgb_to_xyz"]
        out_list = ["matrix"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.custom_text = dpg.add_text("",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        dpg.set_value(self.custom_text,inp[1])
        return inp
    
#####################################################
#####################################################

def xyz_to_sRGB(channels,*argss):
    image_sRGB = colour.XYZ_to_sRGB(channels)
    return image_sRGB,

class XYZ_to_sRGB(NodeV2):
    Title = "XYZ_to_sRGB"
    def __init__(self):
        f = xyz_to_sRGB
        inp_list = ["color"]
        out_list = ["color"]
        super().__init__(f, inp_list, out_list, self.Title)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        return inp

#####################################################
#####################################################

def set_texture_data(width,height,n_channels,channels):
    # PREPARE FOR TEXTURE
    if n_channels==1:   # gray
        l = channels[0].ravel()
        A = np.ones(len(l),dtype=np.float16)
        texture_data = np.array(list(zip(l,l,l,A))).ravel().astype(np.float32)
        
    elif n_channels==3: # RGB
        R = channels[0].ravel()
        G = channels[1].ravel()
        B = channels[2].ravel()
        A = np.ones(len(B),dtype=np.float16)
        texture_data = np.array(list(zip(R,G,B,A))).ravel().astype(np.float32)
        
    else:               # RGBA
        R = channels[0].ravel()
        G = channels[1].ravel()
        B = channels[2].ravel()
        A = channels[3].ravel()
        texture_data = np.array(list(zip(R,G,B,A))).ravel().astype(np.float32)

    print("TEXTURE DATA")
    return texture_data, width, height



def interactive_plot(channels_in,static_texture_ID,plot_y_ID,x_axes,y_axes,*argss):

    # functions 
    channels = channels_in
    height     = channels.shape[0]
    width      = channels.shape[1]
    n_channels = channels.shape[2] if channels.ndim == 3 or channels.ndim == 4 else 1
    print(n_channels,channels.shape)
    
    if n_channels == 1:
        channels= np.array([channels])
    elif n_channels == 3:
        channels= np.array([channels[:,:,0],channels[:,:,1],channels[:,:,2]]  )
    else:
        channels= np.array([channels[:,:,0],channels[:,:,1],channels[:,:,2],channels[:,:,3]] )

    # Nueva imagen a plottear set texture data
    new_texture,width,height = set_texture_data(width, height, n_channels, channels)
    print("refreshing texture")

    # STATIC
    dpg.delete_item(static_texture_ID)
    dpg.delete_item(plot_y_ID)
    print("deleted")
    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=new_texture,tag=static_texture_ID)                    
        print("static_texture")
    dpg.add_image_series(static_texture_ID, [0, 0], [width, height], parent=y_axes,tag=plot_y_ID)
    
    print("refreshed")
    dpg.fit_axis_data(x_axes)
    dpg.fit_axis_data(y_axes)

    return channels_in,

class Interactive_Plot(NodeV2):
    Title = "Interactive Plot"
    def __init__(self):
        f = interactive_plot
        inp_list = ["color"]
        out_list = ["color"]
        super().__init__(f, inp_list, out_list, self.Title)

        with dpg.texture_registry(show=False):
            self.static_texture = dpg.add_static_texture(width=100, height=100, default_value=np.zeros((10000*4))) 
        with dpg.plot(height=108*2, width=192*2, equal_aspects=True,parent=self.static) as self.plotID:
            # REQUIRED: create x and y axes
            self.x_axes = dpg.add_plot_axis(dpg.mvXAxis,no_tick_labels = True, )
                        
            with dpg.plot_axis(dpg.mvYAxis,no_tick_labels = True) as self.y_axes:
                self.plot_y_ID = dpg.add_image_series(self.static_texture, [0, 0], [100, 100], parent=self.y_axes)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        # channels_in,static_texture_ID,plot_y_ID,x_axes,y_axes
        inp.append(self.static_texture)
        inp.append(self.plot_y_ID)
        inp.append(self.x_axes)
        inp.append(self.y_axes)
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

def resize(img,scale_percent,*argss):
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Resize the image to the new dimensions
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img,

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
    x = np.array(x)
    return s_y * (x / (x + s_x))*p,

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
        dpg.set_value(self.line_series, [np.linspace(0,1,1000), curve])

        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(s_y)
        inp.append(s_x)
        inp.append(p)
        return inp
    
#####################################################
#####################################################

def histogram(img,*argss):
    return img,

class Histogram(NodeV2):
    Title = "Histogram"
    def __init__(self):
        f = histogram
        inp_list = ["img"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)
        
        with dpg.plot(label="Histogram", 
                        height=150,
                        width=300,
                        no_title=True,
                        # equal_aspects=True,
                        parent=self.static,
            ) as  self.plot:

            dpg.add_plot_legend()
            # REQUIRED: create x and y axes
            self.x =      dpg.add_plot_axis(dpg.mvXAxis,no_tick_marks=True,no_tick_labels=False)
            self.series = dpg.add_plot_axis(dpg.mvYAxis,no_tick_marks=True,no_tick_labels=True)
            self.line_series1 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="red",parent=self.series)
            self.line_series2 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="green",parent=self.series)
            self.line_series3 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="blue",parent=self.series)

    def recollect_inputs_frombackawrd_nodes(self):
        print("recollecting")
        inp = super().recollect_inputs_frombackawrd_nodes()
        channels = inp[0]
        height     = channels.shape[0]
        width      = channels.shape[1]
        n_channels = channels.shape[2] if channels.ndim == 3 or channels.ndim == 4 else 1

        print("n_channels: ",n_channels)
        res = 300
        if n_channels != 1:
            dpg.show_item(self.line_series2)
            dpg.show_item(self.line_series3)
            ch=channels[:,:,0]
            frecuencias, bordes = np.histogram(ch,bins=res)
            frecuencias = np.array(frecuencias,dtype=np.float32)
            bordes = np.array(bordes,dtype=np.float32)
            dpg.set_value(self.line_series1, [bordes[:-1],frecuencias])

            ch=channels[:,:,1]
            frecuencias, bordes = np.histogram(ch,bins=res)
            frecuencias = np.array(frecuencias,dtype=np.float32)
            bordes = np.array(bordes,dtype=np.float32)
            dpg.set_value(self.line_series2, [bordes[:-1],frecuencias])

            ch=channels[:,:,2]
            frecuencias, bordes = np.histogram(ch,bins=res)
            frecuencias = np.array(frecuencias,dtype=np.float32)
            bordes = np.array(bordes,dtype=np.float32)
            dpg.set_value(self.line_series3, [bordes[:-1],frecuencias])

        else:
            frecuencias, bordes = np.histogram(channels,bins=res)
            frecuencias = np.array(frecuencias,dtype=np.float32)
            bordes = np.array(bordes,dtype=np.float32)
            dpg.set_value(self.line_series1, [bordes[:-1],frecuencias])
            dpg.hide_item(self.line_series2)
            dpg.hide_item(self.line_series3)

        dpg.fit_axis_data(self.x)
        dpg.fit_axis_data(self.series)
        return inp
    
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
        self.min = dpg.add_input_int(label="min",default_value=0,width=120,parent=self.static,callback=self.node_modified)
        self.max = dpg.add_input_int(label="max",default_value=1,width=120,parent=self.static,callback=self.node_modified)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.min))
        inp.append(dpg.get_value(self.max))
        return inp

from modules.father_class import NodeV2
from modules.color_transform import ocio_fun
from modules import INSPECTOR_WINDOW, OCIO_CONFIG, NODE_WINDOW, OCIO_PATH, OCIO
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import colour
# import cv2
matplotlib.use('Agg')

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

def display_fun(img, display, view):
    global OCIO_CONFIG

    img = img.astype(np.float32)
    processor = OCIO_CONFIG.getProcessor(OCIO.ROLE_SCENE_LINEAR, display, view, OCIO.TRANSFORM_DIR_FORWARD)
    cpu = processor.getDefaultCPUProcessor()
    cpu.applyRGB(img)
    
    return img,

def interactive_plot(channels_in,static_texture_ID,plot_y_ID,x_axes,y_axes, display, view,*argss):
    print("INTERACTIVE PLOT",display, view)
    channels = channels_in.copy()

    img = channels.copy()
    # SIZE
    height     = channels.shape[0]
    width      = channels.shape[1]
    n_channels = channels.shape[2] if channels.ndim == 3 or channels.ndim == 4 else 1
    print(n_channels,channels.shape)
    

    # COLOR MANAGEMENT
    if n_channels == 1:
        channels = display_fun( np.dstack((channels_in.copy(), channels_in.copy(), channels_in.copy())) , display, view)[0][:,:,0]
    else:
        channels = display_fun(channels_in.copy(), display, view)[0]

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
    
    return img,

def save_image(path,data,*args):
    print("saving image")
    im = Image.fromarray((data[0] * 255).astype(np.uint8))
    im.save(path)
    print("image saved!!")

    
class Interactive_Plot(NodeV2):
    Title = "Interactive Plot"
    input_space = "sRGB"
    display_list = sorted(OCIO_CONFIG.getDisplays())
    view_list = sorted(OCIO_CONFIG.getViews(display_list[0]))
    display = display_list[0]
    view = view_list[0]

    display_combo = None

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

        dpg.add_button(label="save",parent=self.static,callback=lambda: save_image("test.png",self.recollect_inputs_frombackawrd_nodes()))


    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        
        self.input_space = dpg.get_value(self.combo)
        ocio_gray_line(self.input_space, self)

        processor = OCIO_CONFIG.getProcessor(self.input_space, OCIO.ROLE_REFERENCE)
        cpu = processor.getDefaultCPUProcessor()
        # cpu = processor.createGroupTransform().appendTransform(OCIO_CONFIG.getLooks()[0])
        img = [.5, .5, .5]
        img = cpu.applyRGB(img)
        dpg.set_value(self.middlegray,f"Middle Gray: {img[0]:.4f}")
        
        
        print(self.display)
        if dpg.get_value(self.display_combo) != self.display:

            self.display = dpg.get_value(self.display_combo)
            dpg.delete_item(self.view_combo)

            index_ = self.display_list.index(dpg.get_value(self.display_combo))
            self.view_list = sorted(OCIO_CONFIG.getViews(self.display_list[index_]))
            self.view = self.view_list[0]

            self.view_combo = dpg.add_combo(
                self.view_list,
                tag=self.view_combo,
                label="view",
                default_value=self.view,
                callback=self.node_modified,
                width=100,
                parent=self.tree_node,
                before=self.simpleplot,
            )  
        # self.view = dpg.get_value(self.view_combo)

        # channels_in,static_texture_ID,plot_y_ID,x_axes,y_axes
        inp.append(self.static_texture)
        inp.append(self.plot_y_ID)
        inp.append(self.x_axes)
        inp.append(self.y_axes)
        inp.append(dpg.get_value(self.display_combo))
        inp.append(dpg.get_value(self.view_combo))
        return inp
    
    def custom_node_info(self):
        super().custom_node_info()
        colorSpaceNames = [ cs.getName() for cs in OCIO_CONFIG.getColorSpaces() ]
        # self.view_list = OCIO_CONFIG.getViews(self.display_list[self.display_list.index(dpg.get_value(self.display_combo))])
        

        with dpg.tree_node(label="Node Info",  default_open=True) as self.tree_node:
            self.combo = dpg.add_combo(
                colorSpaceNames,
                label="COLOR_SPACE",
                default_value=self.input_space,
                callback=self.node_modified,
                width=100,
            )  
            self.display_combo = dpg.add_combo(
                self.display_list,
                label="Display",
                default_value=self.display,
                callback=self.node_modified,
                width=100,
            )  
            self.view_combo = dpg.add_combo(
                self.view_list,
                label="view",
                default_value=self.view,
                callback=self.node_modified,
                width=100,
            )  
            print("Display list ",self.display_list)
        

            self.simpleplot = dpg.add_simple_plot(
                default_value=np.linspace(0,5,1000),
                width=275,height=150,
            )
            self.middlegray = dpg.add_text("Middle Gray:")


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

# def line_plot(img,*argss):
#     return img,

# class Line_Plot(NodeV2):
#     Title = "Histogram"
#     def __init__(self):
#         f = line_plot
#         inp_list = ["canal"]
#         out_list = ["canal"]
#         super().__init__(f, inp_list, out_list, self.Title)
        
#         with dpg.plot(label="Histogram", 
#                         height=150,
#                         width=300,
#                         no_title=True,
#                         # equal_aspects=True,
#                         parent=self.static,
#             ) as  self.plot:

#             dpg.add_plot_legend()
#             # REQUIRED: create x and y axes
#             self.x =      dpg.add_plot_axis(dpg.mvXAxis,no_tick_marks=True,no_tick_labels=False)
#             self.series = dpg.add_plot_axis(dpg.mvYAxis,no_tick_marks=True,no_tick_labels=True)
#             self.line_series1 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="red",parent=self.series)
#             self.line_series2 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="green",parent=self.series)
#             self.line_series3 = dpg.add_line_series(np.linspace(0,3,1000),np.linspace(0,1,1000),label="blue",parent=self.series)

#     def recollect_inputs_frombackawrd_nodes(self):
#         print("recollecting")
#         inp = super().recollect_inputs_frombackawrd_nodes()
#         channels = inp[0]
#         height     = channels.shape[0]
#         width      = channels.shape[1]
#         n_channels = channels.shape[2] if channels.ndim == 3 or channels.ndim == 4 else 1

#         print("n_channels: ",n_channels)
#         res = 300
#         if n_channels != 1:
#             dpg.show_item(self.line_series2)
#             dpg.show_item(self.line_series3)
#             ch=channels[:,:,0]
#             frecuencias, bordes = np.histogram(ch,bins=res)
#             frecuencias = np.array(frecuencias,dtype=np.float32)
#             bordes = np.array(bordes,dtype=np.float32)
#             dpg.set_value(self.line_series1, [bordes[:-1],frecuencias])

#             ch=channels[:,:,1]
#             frecuencias, bordes = np.histogram(ch,bins=res)
#             frecuencias = np.array(frecuencias,dtype=np.float32)
#             bordes = np.array(bordes,dtype=np.float32)
#             dpg.set_value(self.line_series2, [bordes[:-1],frecuencias])

#             ch=channels[:,:,2]
#             frecuencias, bordes = np.histogram(ch,bins=res)
#             frecuencias = np.array(frecuencias,dtype=np.float32)
#             bordes = np.array(bordes,dtype=np.float32)
#             dpg.set_value(self.line_series3, [bordes[:-1],frecuencias])

#         else:
#             frecuencias, bordes = np.histogram(channels,bins=res)
#             frecuencias = np.array(frecuencias,dtype=np.float32)
#             bordes = np.array(bordes,dtype=np.float32)
#             dpg.set_value(self.line_series1, [bordes[:-1],frecuencias])
#             dpg.hide_item(self.line_series2)
#             dpg.hide_item(self.line_series3)

#         dpg.fit_axis_data(self.x)
#         dpg.fit_axis_data(self.series)
#         return inp
    
#####################################################
#####################################################


def ocio_gray_line(color_space_in,self,*args):
    img = np.linspace(0, 20, 1000)
    img = np.vstack((img, img, img)).T.reshape(1, 1000, 3)
    
    out = ocio_fun(img, OCIO.ROLE_REFERENCE, color_space_in)[0]
    val = out[0,:,0].astype(float)
    print("OCIO.ROLE_REFERENCE", OCIO.ROLE_REFERENCE)

    dpg.set_value( self.simpleplot, val)

    return img,

class Ocio_Gray_Line(NodeV2):
    Title = "ocio_gray_line"
    def __init__(self):
        f = ocio_gray_line
        inp_list = []
        out_list = []
        super().__init__(f, inp_list, out_list, self.Title)

        # Obtiene el número de espacios de color definidos en la configuración
        config = OCIO_CONFIG

        # Itera sobre todos los espacios de color e imprime sus nombres
        colorSpaceNames = [ cs.getName() for cs in config.getColorSpaces() ]

        self.colour_space_name = dpg.add_combo(
            colorSpaceNames,
            label="COLOR_SPACE",
            parent=self.static,
            default_value="sRGB",
            callback=self.node_modified,
            width=100,
        )
        self.simpleplot = dpg.add_simple_plot(
            parent=self.static,
            default_value=np.linspace(0,20,1000),
            width=300,height=200,
        )


    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append( dpg.get_value(self.colour_space_name) )
        inp.append( self )
        return inp
#####################################################
#####################################################

def chromaticity_diagram(img,color_profile,*args):
    # img = img.astype(np.float64)
    img = img[:,:,:3]
    shape = img.shape
    img = img.reshape(shape[0]*shape[1],3)
    print("EL TAMAÑO DE LA IMAGEN EN EL DIAGRAMA ES\n",img.shape)
    fig, ax = colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
        img, 
        colourspace=color_profile,
        scatter_kwargs={'c': 'k', 'marker': '.'});

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    

    return data/255,

class Chromaticity_Diagram(NodeV2):
    Title = "chromaticity_diagram"
    def __init__(self):
        f = chromaticity_diagram
        inp_list = ["img"]
        out_list = ["chromatic diagram"]
        super().__init__(f, inp_list, out_list, self.Title)

        colorSpaceNames = sorted(colour.RGB_COLOURSPACES)

        self.colour_space_name = dpg.add_combo(
            colorSpaceNames,
            label="COLOR_SPACE",
            parent=self.static,
            default_value="sRGB",
            callback=self.node_modified,
            width=100,
        )


    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append( dpg.get_value(self.colour_space_name) )
        return inp
#####################################################
#####################################################



####
####
from modules import PLOTTERS, NODE_WINDOW_MENU
from modules.interaction import register
# REGISTRO DE NODOS
with dpg.menu(label=PLOTTERS, tag=PLOTTERS, parent=NODE_WINDOW_MENU):
    pass

register_list = [
    Interactive_Plot,
    Histogram, 
    # Ocio_Gray_Line,
    Chromaticity_Diagram,
    
    
    ] #falta resize
for node in register_list:
    register(node, PLOTTERS)
####
####
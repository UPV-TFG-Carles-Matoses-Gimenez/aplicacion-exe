import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2
import os
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import numpy as np
from PIL import Image
import rawpy
import imageio
import colour

from modules import NODE_WINDOW_MENU, LOADDERS, OCIO_CONFIG, OCIO, INSPECTOR_WINDOW
from modules.interaction import register
print("importing loadders module")


#####################################################
#####################################################
import pyroexr

def open_exr_func(path):
    image = pyroexr.load(path)
    imagen = np.dstack((image.channel("R"), image.channel("G"), image.channel("B")))
    return imagen

def load_image(path,space_input="sRGB", *args):
    """get:
    channels [R,G,B]
    """
    print("load image con path: ",path)
    if path != None:
        # extension 
        extension = os.path.splitext(path)[1].lower()
        print(extension)

        # LOAD IMAGE 
        if extension == ".exr":
            img = open_exr_func(path)
        elif extension in [".dpx", ".hdr"]:
            print("Loadding with imageio")
            # imageio.plugins.freeimage.download()
            img = imageio.imread(path)
        else:
            # PROCESO DE IMAGEN
            img = Image.open(path) 
            img = img.convert("RGB")
            img = np.array(img)
        
        # Comprobar la carga 
        if img is None:
            print("No se ha podido leer la imagen")
            return None,

        # normalizar imagenes
        if img.dtype == np.uint8:
            img = np.array(img,dtype=np.float32)
            img /= 255


        # ENDING REFRESH FUNCTIONS
        print("image_refresh, set loadder output")
        print("img shape=",img.shape)

        # COLOR MANAGEMENT
        # Gamma
        # Primaries
        
        processor = OCIO_CONFIG.getProcessor(space_input,OCIO.ROLE_SCENE_LINEAR)
        cpu = processor.getDefaultCPUProcessor()
        cpu.applyRGB(img)
        img = np.array(img,dtype=np.float16)
        return img,

class Load_Image(NodeV2):
    Title = "Load image"
    path = None
    input_space = "sRGB"
    combo = None
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
            modal = True,
            callback=load_image_callback, 
            # cancel_callback=load_image_cancel_callback
            ) as file_dialog:

            dpg.add_file_extension(".*")
            dpg.add_file_extension(".png", 
                                color=(0, 255, 0, 255), 
                                custom_text="[png]")
        # FILE DIALOG #############

        dpg.add_button(parent=self.static,
                       label="load image",
                       callback=lambda:dpg.show_item(file_dialog)
                       )
        
        self.custom_text = dpg.add_text("img...",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        dpg.set_value(self.custom_text,self.split_string(self.path,25))
        if self.combo != None:
            self.input_space = dpg.get_value(self.combo)
        return self.path, self.input_space,

    def custom_node_info(self):
        super().custom_node_info()

        colorSpaceNames = [ cs.getName() for cs in OCIO_CONFIG.getColorSpaces() ]
        with dpg.tree_node(label="Info", default_open=True):
            self.combo = dpg.add_combo(
                colorSpaceNames,
                label="COLOR_SPACE",
                default_value=self.input_space,
                callback=self.node_modified,
                width=100,
            )

        return

#####################################################
#####################################################

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

    return matrix_blacklevel, matrix_bayer, wb, rgb_xyz_matrix, white_level, pt, black,

class Load_Image_Raw(NodeV2):
    Title = "Load image Raw"
    path = None
    def __init__(self):
        f = load_image_raw
        inp_list = []
        out_list = ["matrix","matrix_bayer","wb","rgb_xyz_matrix","white_level","pattern","black level"]
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

    def recollect_inputs_frombackawrd_nodes(self):
        dpg.set_value(self.custom_text,self.split_string(f"img: {self.path}",25))
        return self.path,

#####################################################
#####################################################

def QuadImageGenerator(color_space, length, gradient, cube_size, increment):

    print(color_space, length, gradient, cube_size, increment)
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

    
    # CREACION DE ARRAYS
    # Luz = np.linspace(0,1,length) * max_light
    start_light = 0.001
    Luz = [start_light:= start_light*increment for _ in range(length)]
    color = np.ones(length)*Luz
    negro = np.zeros(length)

    # Para 12 valores (6 primarios) multiple = 2
    multiple = gradient
    incrementer = np.linspace(0,1,multiple+1)
    decrementer = np.linspace(1,0,multiple+1)

    img = []
    # RED
    img.append( np.array(list(zip(color,negro,negro))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(color,color*incrementer[i],negro))).reshape(1,length,3)[0])

    # YELLOW
    img.append( np.array(list(zip(color,color,negro))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(color*decrementer[i],color,negro))).reshape(1,length,3)[0])

    # GREEN
    img.append( np.array(list(zip(negro,color,negro))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(negro,color,color*incrementer[i]))).reshape(1,length,3)[0])

    # CYAN
    img.append( np.array(list(zip(negro,color,color))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(negro,color*decrementer[i],color))).reshape(1,length,3)[0])

    # BLUE
    img.append( np.array(list(zip(negro,negro,color))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(color*incrementer[i],negro,color))).reshape(1,length,3)[0])

    # MAGENTA
    img.append( np.array(list(zip(color,negro,color))).reshape(1,length,3)[0])
    for i in range(1,multiple):
        img.append( np.array(list(zip(color,negro,color*decrementer[i]))).reshape(1,length,3)[0])

    img = np.array(img,dtype=np.float32)
    processor = OCIO_CONFIG.getProcessor(color_space, OCIO.ROLE_SCENE_LINEAR)
    cpu = processor.getDefaultCPUProcessor()
    cpu.applyRGB(img)

    rgb = convert_pixels_to_squares(np.array(img),cube_size)
    return rgb,



class Quad_Image(NodeV2):
    Title = "Quad_Image"
    def __init__(self):
        f = QuadImageGenerator
        inp_list = []
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)

        # Itera sobre todos los espacios de color e imprime sus nombres
        colorSpaceNames = [ cs.getName() for cs in OCIO_CONFIG.getColorSpaces() ]

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
        # length, gradient, cube_size, increment
        inp.append(dpg.get_value(self.colour_space_name))
        inp.append(dpg.get_value(self.length))
        inp.append(dpg.get_value(self.gradient))
        inp.append(dpg.get_value(self.cube_size))
        inp.append(dpg.get_value(self.increment))
        return inp
    
    def custom_node_info(self):
        super().custom_node_info()
        
        self.length =    dpg.add_input_int(label  = "length",callback=self.node_modified,default_value=20)
        self.gradient =  dpg.add_input_int(label  = "gradient",callback=self.node_modified,default_value=2)
        self.cube_size = dpg.add_input_int(label  = "cube_size",callback=self.node_modified,default_value=3)
        self.increment = dpg.add_input_float(label = "increment",callback=self.node_modified,default_value=2)

#####################################################
#####################################################
from colour import LUT3D, LUT1D

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

class Spi3D(NodeV2):
    Title = "LUT_Gen"
    methods = {"LUT3D":LUT3D, "LUT1D":LUT1D}
    def __init__(self):
        f = lut_generator
        inp_list = []
        out_list = ["LUT3D","plot"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.method = dpg.add_combo(
            list(self.methods.keys()), 
            label="method",
            callback=lambda: self.custom_input(dpg.get_value(self.method),self.static,self.node_modified),
            parent=self.static,
            width=200,
            default_value=list(self.methods.keys())[0])
        self.val_id = []

    def custom_input(self,str,parent_id,callback):
        for i in self.val_id:
            dpg.delete_item(i)
        self.val_id = []

        if str == "LUT3D":
            self.val_id.append(dpg.add_input_int(label="size",width=120,parent=parent_id,callback=callback))
        elif str == "LUT1D":
            self.val_id.append(dpg.add_input_float(label="size",width=120,parent=parent_id,callback=callback))

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(self.methods[dpg.get_value(self.method)])
        dictionary = {}

        for i in self.val_id:
            key = dpg.get_item_label(i)
            val = dpg.get_value(i)
            dictionary[key] = val
        inp.append(dictionary)
        print(inp)
        return inp

    def custom_node_info(self):
        super().custom_node_info()

        help = """
        LUT:
            spi1D: Unidimensional (1D) LUT en formato ".spi1d".
            spi3D: Tridimensional (3D) LUT en formato ".spi3d".
            cube: LUT en formato ".cube".
            3dl: LUT en formato ".3dl".
            csp: LUT en formato ".csp".
            mga: LUT en formato ".mga".
            cube (variante de 1D): LUT en formato ".cube".
            lut (LUT en formato de texto plano): LUT en formato ".lut".
            mga (variante de 3D): LUT en formato ".mga".
            spimtx: LUT en formato ".spimtx".
            haldclut (Hald Color Lookup Table): LUT en formato ".png" o ".jpg" que representa una tabla de búsqueda de color en formato de cuadrícula.

        Look:
            clf: Look en formato ".clf" (Common LUT Format).
            look: Look en formato ".look".
            lookml: Look en formato ".lookml".
            3dl (variante de 3D): Look en formato ".3dl".
            lookml (variante de 3D): Look en formato ".lookml".
            look (variante de 1D): Look en formato ".look".
            mga (variante de Look): Look en formato ".mga".
            csp (variante de Look): Look en formato ".csp".
            look (variante de 3D): Look en formato ".look".
        """
        
        print(self.methods)

#####################################################
#####################################################


print("Succes!!: loaders module")

####
####
# REGISTRO DE NODOS
with dpg.menu(label=LOADDERS, tag=LOADDERS,parent=NODE_WINDOW_MENU):
    pass

register_list = [Load_Image,Load_Image_Raw,Quad_Image,Spi3D]
for node in register_list:
    register(node, LOADDERS)

####
####

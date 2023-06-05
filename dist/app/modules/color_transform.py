import dearpygui.dearpygui as dpg
from modules.father_class import NodeV2 
from modules import OCIO, OCIO_CONFIG
import colour
import numpy as np

illuminants = {
    'A': [ 0.44758,  0.40745], 
    'B': [ 0.34842,  0.35161], 
    'C': [ 0.31006,  0.31616], 
    'D50':  [ 0.3457,  0.3585], 
    'D55':  [ 0.33243,  0.34744], 
    'D60':  [ 0.32161671,  0.33761992], 
    'D65':  [ 0.3127,  0.329 ], 
    'D75':  [ 0.29903,  0.31488], 
    'E':  [ 0.33333333,  0.33333333], 
    'FL1':  [ 0.3131,  0.3371], 
    'FL2':  [ 0.3721,  0.3751], 
    'FL3':  [ 0.4091,  0.3941], 
    'FL4':  [ 0.4402,  0.4031], 
    'FL5':  [ 0.3138,  0.3452], 
    'FL6':  [ 0.3779,  0.3882], 
    'FL7':  [ 0.3129,  0.3292], 
    'FL8':  [ 0.3458,  0.3586], 
    'FL9':  [ 0.3741,  0.3727], 
    'FL10':  [ 0.3458,  0.3588], 
    'FL11':  [ 0.3805,  0.3769], 
    'FL12':  [ 0.437 ,  0.4042], 
    'FL3.1':  [ 0.4407,  0.4033], 
    'FL3.2':  [ 0.3808,  0.3734], 
    'FL3.3':  [ 0.3153,  0.3439], 
    'FL3.4':  [ 0.4429,  0.4043], 
    'FL3.5':  [ 0.3749,  0.3672], 
    'FL3.6':  [ 0.3488,  0.36  ], 
    'FL3.7':  [ 0.4384,  0.4045], 
    'FL3.8':  [ 0.382 ,  0.3832], 
    'FL3.9':  [ 0.3499,  0.3591], 
    'FL3.10':  [ 0.3455,  0.356 ], 
    'FL3.11':  [ 0.3245,  0.3434], 
    'FL3.12':  [ 0.4377,  0.4037], 
    'FL3.13':  [ 0.383 ,  0.3724], 
    'FL3.14':  [ 0.3447,  0.3609], 
    'FL3.15':  [ 0.3127,  0.3288], 
    'HP1':  [ 0.533,  0.415], 
    'HP2':  [ 0.4778,  0.4158], 
    'HP3':  [ 0.4302,  0.4075], 
    'HP4':  [ 0.3812,  0.3797], 
    'HP5':  [ 0.3776,  0.3713], 
    'LED-B1':  [ 0.456 ,  0.4078], 
    'LED-B2':  [ 0.4357,  0.4012], 
    'LED-B3':  [ 0.3756,  0.3723], 
    'LED-B4':  [ 0.3422,  0.3502], 
    'LED-B5':  [ 0.3118,  0.3236], 
    'LED-BH1':  [ 0.4474,  0.4066], 
    'LED-RGB1':  [ 0.4557,  0.4211], 
    'LED-V1':  [ 0.4548,  0.4044], 
    'LED-V2':  [ 0.3781,  0.3775], 
    'ID65':  [ 0.31065663,  0.33066309], 
    'ID50':  [ 0.34321137,  0.36020754], 
    'ACES':  [ 0.32168,  0.33767], 
    'Blackmagic Wide Gamut':  [ 0.312717 ,  0.3290312], 
    'DCI-P3':  [ 0.314,  0.351], 
    'ICC D50':  [ 0.34570291,  0.3585386 ], 
    'ISO 7589 Photographic Daylight':  [ 0.3320391 ,  0.34726389], 
    'ISO 7589 Sensitometric Daylight':  [ 0.33381831,  0.35343623], 
    'ISO 7589 Studio Tungsten':  [ 0.43094409,  0.40358544], 
    'ISO 7589 Sensitometric Studio Tungsten':  [ 0.43141822,  0.40747144], 
    'ISO 7589 Photoflood': [ 0.41114602,  0.39371938], 
    'ISO 7589 Sensitometric Photoflood':  [ 0.41202478,  0.39817741], 
    'ISO 7589 Sensitometric Printer': [ 0.41208797,  0.42110498]
 }


#####################################################
#####################################################
def inverse_matrix(rgb_xyz_matrix,*argss):
    RGB_to_XYZ_matrix = np.linalg.inv(rgb_xyz_matrix) #Matriz del archivo raw M^-1 a M
    
    return RGB_to_XYZ_matrix,

class Inverse_Matrix(NodeV2):
    Title = "Inverse_Matrix"
    def __init__(self):
        f = inverse_matrix
        inp_list = ["rgb_to_xyz"]
        out_list = ["inv rgb_to_xyz"]
        super().__init__(f, inp_list, out_list, self.Title)
#####################################################
#####################################################

def white_point(val,*argss):
    return colour.xy_to_XYZ(illuminants[val]),

class White_Point(NodeV2):
    Title = "White Point"
    def __init__(self):
        f = white_point
        inp_list = []
        out_list = ["wp"]
        super().__init__(f, inp_list, out_list, self.Title)
        self.type = dpg.add_combo(
            list(illuminants.keys()), 
            label="wp",
            callback=self.node_modified,
            parent=self.static,
            width=150,
            default_value="D65")

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.type))
        return inp
    
#####################################################
#####################################################

def rgb_add_White_Point(A,B,*argss):
    X = np.linalg.solve(A, B) 
    A[:,0]=A[:,0]*X[0]
    A[:,1]=A[:,1]*X[1]
    A[:,2]=A[:,2]*X[2]

    return A,

class RGB_add_White_Point(NodeV2):
    Title = "RGB_add_White_Point"
    def __init__(self):
        f = rgb_add_White_Point
        inp_list = ["rgb_to_xyz","White Point"]
        out_list = ["rgb_to_xyz"]
        super().__init__(f, inp_list, out_list, self.Title)

#####################################################
#####################################################


def rgb_to_xyz(matrix,rgb_xyz_matrix,illuminant1,illuminant2,*argss):

    new_channels = colour.RGB_to_XYZ(
        #array
        matrix, 
        # Input color space
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant1],
        # Iluminant
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant2],
        rgb_xyz_matrix,
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

        self.illuminant1 = dpg.add_combo(
            list(illuminants.keys()),
            label="from RGB iluminant",
            parent= self.static,
            default_value="D65",
            callback=self.node_modified,
            width=100,
        )
        self.illuminant2 = dpg.add_combo(
            list(illuminants.keys()),
            label="to RGB iluminant",
            parent=self.static,
            default_value="D65",
            callback=self.node_modified,
            width=100,
        )
        self.il1 = dpg.add_text("",parent=self.static)
        self.il2 = dpg.add_text("",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.illuminant1))
        inp.append(dpg.get_value(self.illuminant2))
        dpg.set_value(self.il1,illuminants[dpg.get_value(self.illuminant1)])
        dpg.set_value(self.il2,illuminants[dpg.get_value(self.illuminant2)])
        return inp
    
#####################################################
#####################################################
def xyz_to_rgb(matrix,matrix_XYZ_to_RGB,cctf_encoding=None,illuminant1="D65",illuminant2="D65",*argss):

    new_channels = colour.XYZ_to_RGB(
        #array
        matrix, 
        # Input color space
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant1],
        # Iluminant
        colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant2],
        matrix_XYZ_to_RGB,
        chromatic_adaptation_transform = "Bradford",
        cctf_encoding=cctf_encoding,
            )
    return new_channels,

class XYZ_to_RGB(NodeV2):
    Title = "XYZ_to_RGB"
    def __init__(self):
        f = xyz_to_rgb
        inp_list = ["matrix","matrix_XYZ_to_RGB","cctf_encoding" ]
        out_list = ["matrix"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.illuminant1 = dpg.add_combo(
            list(illuminants.keys()),
            label="from XYZ iluminant",
            parent= self.static,
            default_value="D65",
            callback=self.node_modified,
            width=100,
        )
        self.illuminant2 = dpg.add_combo(
            list(illuminants.keys()),
            label="to RGB iluminant",
            parent=self.static,
            default_value="D65",
            callback=self.node_modified,
            width=100,
        )
        self.il1 = dpg.add_text("[0.3127, 0.329]",parent=self.static)
        self.il2 = dpg.add_text("[0.3127, 0.329]",parent=self.static)

    def recollect_inputs_frombackawrd_nodes(self):
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(dpg.get_value(self.illuminant1))
        inp.append(dpg.get_value(self.illuminant2))
        dpg.set_value(self.il1,illuminants[dpg.get_value(self.illuminant1)])
        dpg.set_value(self.il2,illuminants[dpg.get_value(self.illuminant2)])
        return inp
    
#####################################################
#####################################################
def xyz_to_sRGB(channels,*argss):
    image_sRGB = colour.XYZ_to_sRGB(channels,apply_cctf_encoding=False)
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

def color_SPACE(string,*argss):
    cs = colour.RGB_COLOURSPACES[string]
    return cs.name,cs.primaries,cs.whitepoint_name,cs.matrix_RGB_to_XYZ,cs.matrix_XYZ_to_RGB,cs.cctf_encoding,cs.cctf_decoding,cs.use_derived_matrix_RGB_to_XYZ,cs.use_derived_matrix_XYZ_to_RGB

class COLOR_SPACE(NodeV2):
    Title = "COLOR_SPACE"
    def __init__(self):
        f = color_SPACE
        inp_list = []
        out_list = ["name","primaries","whitepoint_name","matrix_RGB_to_XYZ","matrix_XYZ_to_RGB","cctf_encoding","cctf_decoding","use_derived_matrix_RGB_to_XYZ","use_derived_matrix_XYZ_to_RGB"]
        super().__init__(f, inp_list, out_list, self.Title)

        self.colour_space_name = dpg.add_combo(
            sorted(colour.RGB_COLOURSPACES),
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

def eotf_RGB(img,_function,*argss):
    return _function(img),

class EOTF_RGB(NodeV2):
    Title = "EOTF_RGB"
    def __init__(self):
        f = eotf_RGB
        inp_list = ["img","eotf"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)


#####################################################
#####################################################

def ocio_fun(img,color_space_in,color_space_out,*args):
    global OCIO_CONFIG
    
    img = img.astype(np.float32)
    processor = OCIO_CONFIG.getProcessor(color_space_in, color_space_out)
    cpu = processor.getDefaultCPUProcessor()
    cpu.applyRGB(img)

    return img,

class Ocio_Management(NodeV2):
    Title = "Ocio_Management"
    def __init__(self):
        f = ocio_fun
        inp_list = ["img"]
        out_list = ["img"]
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
        self.colour_space_name2 = dpg.add_combo(
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
        inp.append( dpg.get_value(self.colour_space_name2) )
        return inp

#####################################################
#####################################################
# MATH FUNCTIONS PARA DEBUGGEAR
def _fabs(b):
    return np.abs(b)
def _powf(a,b):
    return np.power(a,b)
def _logf(a):
    return np.log(a)
def _powf(a,b):
    return np.power(a,b)
def _log2f(a):
    return np.log2(a)
def _sqrt(a):
    return np.sqrt(a)
def _expf(a):
    return np.exp(a)

# // Helper function to create a float3x3
def make_float3x3(a,b,c):
    return np.array([a,b,c])

# // Return identity 3x3 matrix
def identity():
    return np.identity(3)

# // Multiply 3x3 matrix m and float3 vector v
def vdot(m,v):
    return np.array([m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2], m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2], m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]]);

# // Safe division of float a by float b
def sdivf(a,b):
    if _fabs(b) < 1e-4: 
        return 0
    else: 
        return a/b

# // Safe division of float3 a by float b
def sdivf3f(a,b):
    return np.array([sdivf(a[0], b), sdivf(a[1], b), sdivf(a[2], b)])

# // Safe division of float3 a by float3 b
def sdivf3f3(a,b):
    return np.array([sdivf(a[0], b[0]), sdivf(a[1], b[1]), sdivf(a[2], b[2])]);

# // Safe power function raising float a to power float b
def spowf(a,b):
    if a <= 0:
        return a
    else:
        return _powf(a,b)

# // Safe power function raising float3 a to power float b
def spowf3(a,b):
    return np.array([_powf(a[0],b),  _powf(a[1],b),  _powf(a[2],b)])

# // Clamp each component of float3 a to be between float mn and float mx
def clampf3(a,mn,mx):
    return np.clip(a,mn,mx)

def eotf_hlg(rgb,inverse):
    HLG_Lw = 1000
    HLG_Ls = 5
    h_a = 0.17883277
    h_b = 1 - 4*0.17883277
    h_c = 0.5 - h_a*_logf(4*h_a)
    h_g = 1.2*_powf(1.111, _log2f(HLG_Lw/1000))*_powf(0.98, _log2f(np.max(1e-6, HLG_Ls)/5))

    if inverse == 1:
        Yd = 0.2627*rgb[0] + 0.6780*rgb[1] + 0.0593*rgb[2]
        rgb = rgb*_powf(Yd, (1.0 - h_g)/h_g)
        rgb[0] = _sqrt(3 * rgb[0]) if rgb[0] <= 1 / 12 else h_a * _logf(12 * rgb[0] - h_b) + h_c
        rgb[1] = _sqrt(3 * rgb[1]) if rgb[1] <= 1 / 12 else h_a * _logf(12 * rgb[1] - h_b) + h_c
        rgb[2] = _sqrt(3 * rgb[2]) if rgb[2] <= 1 / 12 else h_a * _logf(12 * rgb[2] - h_b) + h_c

    else:
        rgb[0] = rgb[0]*rgb[0]/3 if rgb[0] <= 0.5 else (_expf((rgb[0]-h_c)/h_a)+h_b)/12
        rgb[1] = rgb[1]*rgb[1]/3 if rgb[1] <= 0.5 else (_expf((rgb[1]-h_c)/h_a)+h_b)/12
        rgb[2] = rgb[2]*rgb[2]/3 if rgb[2] <= 0.5 else (_expf((rgb[2]-h_c)/h_a)+h_b)/12
        Ys = 0.2627*rgb[0] + 0.6780*rgb[1] + 0.0593*rgb[2]
        rgb = rgb*_powf(Ys, h_g - 1)

    return rgb

def eotf_pq(rgb,inverse):
    Lp = 1
    m1 = 2610/16384
    m2 = 2523/32
    c1 = 107/128
    c2 = 2413/128
    c3 = 2392/128

    if inverse == 1 :
        rgb /= Lp
        rgb = spowf3(rgb, m1)
        rgb[0] = spowf((c1 + c2*rgb[0])/(1 + c3*rgb[0]), m2)
        rgb[1] = spowf((c1 + c2*rgb[1])/(1 + c3*rgb[1]), m2)
        rgb[2] = spowf((c1 + c2*rgb[2])/(1 + c3*rgb[2]), m2)
    else :
        rgb = spowf3(rgb, 1/m2)
        rgb[0] = spowf((rgb[0] - c1)/(c2 - c3*rgb[0]), 1/m1)*Lp
        rgb[1] = spowf((rgb[1] - c1)/(c2 - c3*rgb[1]), 1/m1)*Lp
        rgb[2] = spowf((rgb[2] - c1)/(c2 - c3*rgb[2]), 1/m1)*Lp
    
    return rgb

def narrow_hue_angles(v):
    return np.array([
        np.clip(v[0]-(v[1]+v[2]),0,2),
        np.clip(v[1]-(v[0]+v[2]),0,2),
        np.clip(v[2]-(v[0]+v[1]),0,2),
    ])

def tonescale(x,m,s,c,invert):
    if invert == 0:
        return spowf(m*x/(x+s),c)
    else:
        ip = 1/c
        return spowf(s*x,ip)/(m-spowf(x,ip))
    
def flare(x,fl,invert):
    if invert == 0:
        return spowf(x,2)/(x+fl)
    else:
        return (x+_sqrt(x*(4*fl+x)))/2

    
def transform(p_R,p_G,p_B,in_gamut="i_rec709", Lp=100, gb=0.12, c=1.1, fl=0.01, rw=0.2, bw=0.25,
                 dch=0.5, dch_toe=0.005, dch_r=-0.5, dch_g=0.0, dch_b=0.0, dch_c=-0.34,
                 dch_m=-0.34, dch_y=0.0, hs_r=0.4, hs_g=-0.1, hs_b=-0.55, v_r=0.8, v_g=0.66,
                 v_b=0.5, v_c=0.66, v_m=0.5, v_y=0.33, display_gamut="Rec709", EOTF="rec1886"):
    # Gamut Conversion Matrices
    matrix_ap0_to_xyz = np.array([[0.93863094875, -0.00574192055, 0.017566898852],
                                [0.338093594922, 0.727213902811, -0.065307497733],
                                [0.000723121511, 0.000818441849, 1.0875161874]])
    matrix_ap1_to_xyz = np.array([[0.652418717672, 0.127179925538, 0.170857283842],
                                [0.268064059194, 0.672464478993, 0.059471461813],
                                [-0.00546992851, 0.005182799977, 1.08934487929]])
    matrix_rec709_to_xyz = np.array([[0.412390917540, 0.357584357262, 0.180480793118],
                                    [0.212639078498, 0.715168714523, 0.072192311287],
                                    [0.019330825657, 0.119194783270, 0.950532138348]])
    matrix_p3d65_to_xyz = np.array([[0.486571133137, 0.265667706728, 0.198217317462],
                                    [0.228974640369, 0.691738605499, 0.079286918044],
                                    [0.000000000000, 0.045113388449, 1.043944478035]])
    matrix_rec2020_to_xyz = np.array([[0.636958122253, 0.144616916776, 0.168880969286],
                                    [0.262700229883, 0.677998125553, 0.059301715344],
                                    [0.000000000000, 0.028072696179, 1.060985088348]])
    matrix_arriwg3_to_xyz = np.array([[0.638007619284, 0.214703856337, 0.097744451431],
                                    [0.291953779, 0.823841041511, -0.11579482051],
                                    [0.002798279032, -0.067034235689, 1.15329370742]])
    matrix_arriwg4_to_xyz = np.array([[0.704858320407, 0.12976029517, 0.115837311474],
                                    [0.254524176404, 0.781477732712, -0.036001909116],
                                    [0.0, 0.0, 1.08905775076]])
    matrix_redwg_to_xyz = np.array([[0.735275208950, 0.068609409034, 0.146571278572],
                                    [0.286694079638, 0.842979073524, -0.129673242569],
                                    [-0.079680845141, -0.347343206406, 1.516081929207]])
    matrix_sonysgamut3_to_xyz = np.array([[0.706482713192, 0.128801049791, 0.115172164069],
                                        [0.270979670813, 0.786606411221, -0.057586082034],
                                        [-0.009677845386, 0.004600037493, 1.094135558]])
    matrix_sonysgamut3cine_to_xyz = np.array([[0.599083920758, 0.248925516115, 0.102446490178],
                                            [0.215075820116, 0.885068501744, -0.100144321859],
                                            [-0.032065849545, -0.027658390679, 1.14878199098]])
    matrix_bmdwg_to_xyz = np.array([[0.606538414955, 0.220412746072, 0.123504832387],
                                    [0.267992943525, 0.832748472691, -0.100741356611],
                                    [-0.029442556202, -0.086612440646, 1.205112814903]])
    matrix_egamut_to_xyz = np.array([[0.705396831036, 0.164041340351, 0.081017754972],
                                    [0.280130714178, 0.820206701756, -0.100337378681],
                                    [-0.103781513870, -0.072907261550, 1.265746593475]])
    matrix_davinciwg_to_xyz = np.array([[0.700622320175, 0.148774802685, 0.101058728993],
                                        [0.274118483067, 0.873631775379, -0.147750422359],
                                        [-0.098962903023, -0.137895315886, 1.325916051865]])
    ###################################
    matrix_xyz_to_rec709 = np.array([[3.2409699419, -1.53738317757, -0.498610760293],
                                    [-0.969243636281, 1.87596750151, 0.041555057407],
                                    [0.055630079697, -0.203976958889, 1.05697151424]])
    matrix_xyz_to_p3d65 = np.array([[2.49349691194, -0.931383617919, -0.402710784451],
                                    [-0.829488969562, 1.76266406032, 0.023624685842],
                                    [0.035845830244, -0.076172389268, 0.956884524008]])
    matrix_xyz_to_rec2020 = np.array([[1.71665118797, -0.355670783776, -0.253366281374],
                                  [-0.666684351832, 1.61648123664, 0.015768545814],
                                  [0.017639857445, -0.042770613258, 0.942103121235]])

    
    if (in_gamut == "i_xyz"): in_to_xyz = identity();
    elif (in_gamut == "i_ap0"): in_to_xyz = matrix_ap0_to_xyz;
    elif (in_gamut == "i_ap1"): in_to_xyz = matrix_ap1_to_xyz;
    elif (in_gamut == "i_p3d65"): in_to_xyz = matrix_p3d65_to_xyz;
    elif (in_gamut == "i_rec2020"): in_to_xyz = matrix_rec2020_to_xyz;
    elif (in_gamut == "i_rec709"): in_to_xyz = matrix_rec709_to_xyz;
    elif (in_gamut == "i_awg3"): in_to_xyz = matrix_arriwg3_to_xyz;
    elif (in_gamut == "i_awg4"): in_to_xyz = matrix_arriwg4_to_xyz;
    elif (in_gamut == "i_rwg"): in_to_xyz = matrix_redwg_to_xyz;
    elif (in_gamut == "i_sgamut3"): in_to_xyz = matrix_sonysgamut3_to_xyz;
    elif (in_gamut == "i_sgamut3cine"): in_to_xyz = matrix_sonysgamut3cine_to_xyz;
    elif (in_gamut == "i_bmdwg"): in_to_xyz = matrix_bmdwg_to_xyz;
    elif (in_gamut == "i_egamut"): in_to_xyz = matrix_egamut_to_xyz;
    elif (in_gamut == "i_davinciwg"): in_to_xyz = matrix_davinciwg_to_xyz;

    if (display_gamut == "Rec709"): xyz_to_display = matrix_xyz_to_rec709;
    elif (display_gamut == "P3D65"): xyz_to_display = matrix_xyz_to_p3d65;
    elif (display_gamut == "Rec2020"): xyz_to_display = matrix_xyz_to_rec2020;
    elif (display_gamut == "None"): xyz_to_display = identity();

    if   (EOTF == "lin"):     eotf = 0;
    elif (EOTF == "srgb"):    eotf = 1;
    elif (EOTF == "rec1886"): eotf = 2;
    elif (EOTF == "dci"):     eotf = 3;
    elif (EOTF == "pq"):      eotf = 4;
    elif (EOTF == "hlg"):     eotf = 5;

    # ds = eotf == 4 ? 0.01f : eotf == 5 ? 0.1f : 100.0f/Lp;
    ds = 0.01 if eotf == 4 else 0.1 if eotf == 5 else 100/Lp
    clamp_max = ds*Lp/100

    # // input scene-linear peak x intercept
    px = 128.0*_logf(Lp)/_logf(100.0) - 64
    # // output display-linear peak y intercept
    py = Lp/100
    # // input scene-linear middle grey x intercept
    gx = 0.18
    # // output display-linear middle grey y intercept
    gy = 11.696/100*(1 + gb*_logf(py)/_logf(2))
    # // s0 and s are input x scale for middle grey intersection constraint
    # // m0 and m are output y scale for peak white intersection constraint
    s0 = flare(gy, fl, 1)
    m0 = flare(py, fl, 1)
    ip = 1/c
    s = (px*gx*(_powf(m0, ip) - _powf(s0, ip)))/(px*_powf(s0, ip) - gx*_powf(m0, ip))
    m = _powf(m0, ip)*(s + px)/px

    #   /* Rendering Code ------------------------------------------ */
    rgb = np.array([p_R, p_G, p_B])
    # // Convert into display gamut
    rgb = vdot(in_to_xyz, rgb)
    rgb = vdot(xyz_to_display, rgb)

    #   // Take the min and the max of rgb. These are used to calculate hue angle, chrominance, and rgb ratios
    mx = np.max(rgb)
    mn = np.min(rgb)

    h_rgb = sdivf3f(rgb - mn, mx)
    h_rgb = narrow_hue_angles(h_rgb)
    h_cmy = sdivf3f(mx - rgb, mx)
    h_cmy = narrow_hue_angles(h_cmy)

    # // chroma here does not refer to any perceptual metric. It is merely the normalized distance from achromatic 
    ch = 1 - sdivf(mn, mx)

    gw = 1 - (rw + bw)
    lum = np.max([1e-5, rgb[0]*rw + rgb[1]*gw + rgb[2]*bw])

    #   // RGB Ratios using luminance as the vector norm
    rats = sdivf3f(rgb, lum)

    # // Apply tonescale function to lum
    ts = tonescale(lum, m, s, c, 0)
    ts = flare(ts, fl, 0)

    dch_s = dch/s
    # // Unpremult hue rgb and hue cmy by chroma
    ha_rgb = sdivf3f(h_rgb, ch)
    ha_cmy = sdivf3f(h_cmy, ch)

    #   // The final dechroma weight to use in the chroma compression function
    dch_w = dch_s*(dch_r*ha_rgb[0] + 1)*(dch_g*ha_rgb[1] + 1)*(dch_b*ha_rgb[2] + 1)*(dch_c*ha_cmy[0] + 1)*(dch_m*ha_cmy[1] + 1)*(dch_y*ha_cmy[2] + 1)

    #   // Chroma compression factor, used to lerp towards 1 in rgb ratios, compressing chroma
    ccf = sdivf(1, lum*dch_w + 1)

    toe_ccf = (dch_toe + 1)*sdivf(lum, lum + dch_toe)*ccf
    val_w = (1 - h_rgb[0]*v_r)*(1 - h_rgb[1]*v_g)*(1 - h_rgb[2]*v_b)*(1 - h_cmy[0]*v_c)*(1 - h_cmy[1]*v_m)*(1 - h_cmy[2]*v_y)
    hs_w = (1 - ccf)*h_rgb

    #   // Apply hue shift to RGB Ratios
    rats = np.array([rats[0] + hs_w[2]*hs_b - hs_w[1]*hs_g, rats[1] + hs_w[0]*hs_r - hs_w[2]*hs_b, rats[2] + hs_w[1]*hs_g - hs_w[0]*hs_r])

    # // Apply chroma compression to RGB Ratios
    rats = 1 - toe_ccf + rats * toe_ccf
        
    # // Apply chroma value normalization to RGB Ratios
    rats_mx = np.max(rats)
    # // Normalized rgb ratios
    rats_n = sdivf3f(rats, rats_mx)
    # // Mix based on chroma value weights factor
    rats = rats_n*(1 - val_w) + rats*val_w

    # // Multiply tonescale by final RGB Ratios
    rgb = rats*ts

    rgb *= ds
    rgb = clampf3(rgb, 0, clamp_max)

    #   // Apply inverse Display EOTF
    eotf_p = 2 + eotf * 0.2
    if (eotf > 0) and (eotf < 4):
        rgb = spowf3(rgb, 1/eotf_p)
    elif (eotf == 4):
        rgb = eotf_pq(rgb, 1)
    elif (eotf == 5):
        rgb = eotf_hlg(rgb, 1)

    return rgb


class openDRT_class:
    # Gamut Conversion Matrices
    matrix_ap0_to_xyz = np.array([[0.93863094875, -0.00574192055, 0.017566898852],
                                [0.338093594922, 0.727213902811, -0.065307497733],
                                [0.000723121511, 0.000818441849, 1.0875161874]])
    matrix_ap1_to_xyz = np.array([[0.652418717672, 0.127179925538, 0.170857283842],
                                [0.268064059194, 0.672464478993, 0.059471461813],
                                [-0.00546992851, 0.005182799977, 1.08934487929]])
    matrix_rec709_to_xyz = np.array([[0.412390917540, 0.357584357262, 0.180480793118],
                                    [0.212639078498, 0.715168714523, 0.072192311287],
                                    [0.019330825657, 0.119194783270, 0.950532138348]])
    matrix_p3d65_to_xyz = np.array([[0.486571133137, 0.265667706728, 0.198217317462],
                                    [0.228974640369, 0.691738605499, 0.079286918044],
                                    [0.000000000000, 0.045113388449, 1.043944478035]])
    matrix_rec2020_to_xyz = np.array([[0.636958122253, 0.144616916776, 0.168880969286],
                                    [0.262700229883, 0.677998125553, 0.059301715344],
                                    [0.000000000000, 0.028072696179, 1.060985088348]])
    matrix_arriwg3_to_xyz = np.array([[0.638007619284, 0.214703856337, 0.097744451431],
                                    [0.291953779, 0.823841041511, -0.11579482051],
                                    [0.002798279032, -0.067034235689, 1.15329370742]])
    matrix_arriwg4_to_xyz = np.array([[0.704858320407, 0.12976029517, 0.115837311474],
                                    [0.254524176404, 0.781477732712, -0.036001909116],
                                    [0.0, 0.0, 1.08905775076]])
    matrix_redwg_to_xyz = np.array([[0.735275208950, 0.068609409034, 0.146571278572],
                                    [0.286694079638, 0.842979073524, -0.129673242569],
                                    [-0.079680845141, -0.347343206406, 1.516081929207]])
    matrix_sonysgamut3_to_xyz = np.array([[0.706482713192, 0.128801049791, 0.115172164069],
                                        [0.270979670813, 0.786606411221, -0.057586082034],
                                        [-0.009677845386, 0.004600037493, 1.094135558]])
    matrix_sonysgamut3cine_to_xyz = np.array([[0.599083920758, 0.248925516115, 0.102446490178],
                                            [0.215075820116, 0.885068501744, -0.100144321859],
                                            [-0.032065849545, -0.027658390679, 1.14878199098]])
    matrix_bmdwg_to_xyz = np.array([[0.606538414955, 0.220412746072, 0.123504832387],
                                    [0.267992943525, 0.832748472691, -0.100741356611],
                                    [-0.029442556202, -0.086612440646, 1.205112814903]])
    matrix_egamut_to_xyz = np.array([[0.705396831036, 0.164041340351, 0.081017754972],
                                    [0.280130714178, 0.820206701756, -0.100337378681],
                                    [-0.103781513870, -0.072907261550, 1.265746593475]])
    matrix_davinciwg_to_xyz = np.array([[0.700622320175, 0.148774802685, 0.101058728993],
                                        [0.274118483067, 0.873631775379, -0.147750422359],
                                        [-0.098962903023, -0.137895315886, 1.325916051865]])
    ###################################
    matrix_xyz_to_rec709 = np.array([[3.2409699419, -1.53738317757, -0.498610760293],
                                    [-0.969243636281, 1.87596750151, 0.041555057407],
                                    [0.055630079697, -0.203976958889, 1.05697151424]])
    matrix_xyz_to_p3d65 = np.array([[2.49349691194, -0.931383617919, -0.402710784451],
                                    [-0.829488969562, 1.76266406032, 0.023624685842],
                                    [0.035845830244, -0.076172389268, 0.956884524008]])
    matrix_xyz_to_rec2020 = np.array([[1.71665118797, -0.355670783776, -0.253366281374],
                                  [-0.666684351832, 1.61648123664, 0.015768545814],
                                  [0.017639857445, -0.042770613258, 0.942103121235]])


    def __init__(self,img,in_gamut="i_rec709", Lp=100, gb=0.12, c=1.1, fl=0.01, rw=0.2, bw=0.25,
                 dch=0.5, dch_toe=0.005, dch_r=-0.5, dch_g=0.0, dch_b=0.0, dch_c=-0.34,
                 dch_m=-0.34, dch_y=0.0, hs_r=0.4, hs_g=-0.1, hs_b=-0.55, v_r=0.8, v_g=0.66,
                 v_b=0.5, v_c=0.66, v_m=0.5, v_y=0.33, display_gamut="Rec709", EOTF="rec1886") -> None:
        
        self.img = img
        self.in_gamut = in_gamut
        self.Lp = Lp
        self.gb = gb
        self.c = c
        self.fl = fl
        self.rw = rw
        self.bw = bw
        self.dch = dch
        self.dch_toe = dch_toe
        self.dch_r = dch_r
        self.dch_g = dch_g
        self.dch_b = dch_b
        self.dch_c = dch_c
        self.dch_m = dch_m
        self.dch_y = dch_y
        self.hs_r = hs_r
        self.hs_g = hs_g
        self.hs_b = hs_b
        self.v_r = v_r
        self.v_g = v_g
        self.v_b = v_b
        self.v_c = v_c
        self.v_m = v_m
        self.v_y = v_y
        self.display_gamut = display_gamut
        self.EOTF = EOTF

    def in_gamut_fun(self, in_gamut):
        # in_to_xyz = np.identity(3)
        if (in_gamut == "i_xyz"): in_to_xyz = np.identity(3);
        elif (in_gamut == "i_ap0"): in_to_xyz = self.matrix_ap0_to_xyz;
        elif (in_gamut == "i_ap1"): in_to_xyz = self.matrix_ap1_to_xyz;
        elif (in_gamut == "i_p3d65"): in_to_xyz = self.matrix_p3d65_to_xyz;
        elif (in_gamut == "i_rec2020"): in_to_xyz = self.matrix_rec2020_to_xyz;
        elif (in_gamut == "i_rec709"): in_to_xyz = self.matrix_rec709_to_xyz;
        elif (in_gamut == "i_awg3"): in_to_xyz = self.matrix_arriwg3_to_xyz;
        elif (in_gamut == "i_awg4"): in_to_xyz = self.matrix_arriwg4_to_xyz;
        elif (in_gamut == "i_rwg"): in_to_xyz = self.matrix_redwg_to_xyz;
        elif (in_gamut == "i_sgamut3"): in_to_xyz = self.matrix_sonysgamut3_to_xyz;
        elif (in_gamut == "i_sgamut3cine"): in_to_xyz = self.matrix_sonysgamut3cine_to_xyz;
        elif (in_gamut == "i_bmdwg"): in_to_xyz = self.matrix_bmdwg_to_xyz;
        elif (in_gamut == "i_egamut"): in_to_xyz = self.matrix_egamut_to_xyz;
        elif (in_gamut == "i_davinciwg"): in_to_xyz = self.matrix_davinciwg_to_xyz;
        return in_to_xyz
    
    def out_gamut_fun(self,display_gamut):
        # xyz_to_display = np.identity(3)
        if (display_gamut == "Rec709"): xyz_to_display = self.matrix_xyz_to_rec709;
        elif (display_gamut == "P3D65"): xyz_to_display = self.matrix_xyz_to_p3d65;
        elif (display_gamut == "Rec2020"): xyz_to_display = self.matrix_xyz_to_rec2020;
        elif (display_gamut == "None"): xyz_to_display = np.identity(3);
        return xyz_to_display
    
    def transform(self):

        in_to_xyz = self.in_gamut_fun(self.in_gamut)
        xyz_to_display = self.out_gamut_fun(self.display_gamut)

        # ds = eotf == 4 ? 0.01f : eotf == 5 ? 0.1f : 100.0f/Lp;
        # ds = 0.01 if eotf == 4 else 0.1 if eotf == 5 else 100/Lp
        ds = 100/self.Lp
        clamp_max = ds*self.Lp/100

        # // input scene-linear peak x intercept
        px = 128.0*_logf(self.Lp)/_logf(100.0) - 64
        # // output display-linear peak y intercept
        py = self.Lp/100
        # // input scene-linear middle grey x intercept
        gx = 0.18
        # // output display-linear middle grey y intercept
        gy = 11.696/100*(1 + self.gb*_logf(py)/_logf(2))
        # // s0 and s are input x scale for middle grey intersection constraint
        # // m0 and m are output y scale for peak white intersection constraint
        s0 = flare(gy, self.fl, 1)
        m0 = flare(py, self.fl, 1)
        ip = 1/self.c
        s = (px*gx*(_powf(m0, ip) - _powf(s0, ip)))/(px*_powf(s0, ip) - gx*_powf(m0, ip))
        m = _powf(m0, ip)*(s + px)/px

        #   /* Rendering Code ------------------------------------------ */
        rgb = self.img.copy()
        # // Convert into display gamut

        rgb = np.dot(rgb.reshape(-1, 3), in_to_xyz)
        rgb = rgb.reshape(self.img.shape)

        rgb = np.dot(rgb.reshape(-1, 3), xyz_to_display)
        rgb = rgb.reshape(self.img.shape)

        #   // Take the min and the max of rgb. These are used to calculate hue angle, chrominance, and rgb ratios
        mx = np.max(rgb,axis=2)
        mn = np.min(rgb,axis=2)

        h_rgb = (rgb-mn[..., None])/mx[..., None]
        h_rgb = np.nan_to_num(h_rgb, nan=0, posinf=0, neginf=0)
        h_rgb[:,:,0] = np.clip(h_rgb[:,:,0] - (h_rgb[:,:,1]+h_rgb[:,:,2]),0,2)
        h_rgb[:,:,1] = np.clip(h_rgb[:,:,1] - (h_rgb[:,:,0]+h_rgb[:,:,2]),0,2)
        h_rgb[:,:,2] = np.clip(h_rgb[:,:,2] - (h_rgb[:,:,1]+h_rgb[:,:,0]),0,2)

        h_cmy = (mx[..., None] - rgb) / mx[..., None]
        h_cmy = np.nan_to_num(h_cmy, nan=0, posinf=0, neginf=0)
        h_cmy[:,:,0] = np.clip(h_cmy[:,:,0] - (h_cmy[:,:,1]+h_cmy[:,:,2]),0,2)
        h_cmy[:,:,1] = np.clip(h_cmy[:,:,1] - (h_cmy[:,:,0]+h_cmy[:,:,2]),0,2)
        h_cmy[:,:,2] = np.clip(h_cmy[:,:,2] - (h_cmy[:,:,1]+h_cmy[:,:,0]),0,2)

        # // chroma here does not refer to any perceptual metric. It is merely the normalized distance from achromatic 
        ch = 1 - (mn/mx)
        ch = np.nan_to_num(ch, nan=0, posinf=0, neginf=0)
        gw = 1 - (self.rw + self.bw)
        temporal_max = np.zeros(self.img.shape[:2])+1e-5
        lum = np.maximum(temporal_max, rgb[:,:,0]*self.rw + rgb[:,:,1]*gw + rgb[:,:,2]*self.bw)

        #   // RGB Ratios using luminance as the vector norm
        rats = rgb / lum[...,None]
        # // Apply tonescale function to lum
        ts = np.power(m*lum/(lum+s),self.c)
        ts = np.power(ts,2)/(ts+self.fl)

        # // // Unused inverse direction
        # ts = (lum + np.sqrt(lum*(4.0*self.fl + lum)))/2.0;
        # ip = 1.0/self.c;
        # ts = np.power(s*lum,ip) / (m-np.power(lum,ip));
        # ts = np.nan_to_num(ts, nan=0, posinf=0, neginf=0)

        dch_s = self.dch/s
        # // Unpremult hue rgb and hue cmy by chroma
        ha_rgb = h_rgb / ch[...,None]
        ha_cmy = h_cmy / ch[...,None]
        ha_rgb = np.nan_to_num(ha_rgb, nan=0, posinf=0, neginf=0)
        ha_cmy = np.nan_to_num(ha_cmy, nan=0, posinf=0, neginf=0)

        #   // The final dechroma weight to use in the chroma compression function
        dch_w = dch_s*(self.dch_r*ha_rgb[:,:,0] + 1)*(self.dch_g*ha_rgb[:,:,1] + 1)*(self.dch_b*ha_rgb[:,:,2] + 1)*(self.dch_c*ha_cmy[:,:,0] + 1)*(self.dch_m*ha_cmy[:,:,1] + 1)*(self.dch_y*ha_cmy[:,:,2] + 1)


        #   // Chroma compression factor, used to lerp towards 1 in rgb ratios, compressing chroma
        ccf = 1 / (lum*dch_w + 1)

        toe_ccf = (self.dch_toe + 1)*(lum /( lum + self.dch_toe))*ccf
        val_w = (1 - h_rgb[:,:,0]*self.v_r)*(1 - h_rgb[:,:,1]*self.v_g)*(1 - h_rgb[:,:,2]*self.v_b)*(1 - h_cmy[:,:,0]*self.v_c)*(1 - h_cmy[:,:,1]*self.v_m)*(1 - h_cmy[:,:,2]*self.v_y)
        hs_w = (1 - ccf[...,None])*h_rgb

        #   // Apply hue shift to RGB Ratios
        # rats = np.array([rats[:,:,0] + hs_w[:,:,2]*self.hs_b - hs_w[:,:,1]*self.hs_g, rats[:,:,1] + hs_w[:,:,0]*self.hs_r - hs_w[:,:,2]*self.hs_b, rats[:,:,2] + hs_w[:,:,1]*self.hs_g - hs_w[:,:,0]*self.hs_r])
        rats[:,:,0] = rats[:,:,0] + hs_w[:,:,2]*self.hs_b - hs_w[:,:,1]*self.hs_g
        rats[:,:,1] = rats[:,:,1] + hs_w[:,:,0]*self.hs_r - hs_w[:,:,2]*self.hs_b
        rats[:,:,2] = rats[:,:,2] + hs_w[:,:,1]*self.hs_g - hs_w[:,:,0]*self.hs_r


        # // Apply chroma compression to RGB Ratios
        rats = 1 - toe_ccf[...,None] + rats * toe_ccf[...,None]

        # // Apply chroma value normalization to RGB Ratios
        rats_mx = np.max(rats,axis=2)
        # // Normalized rgb ratios
        rats_n = rats / rats_mx[...,None]
        # // Mix based on chroma value weights factor
        rats = rats_n*(1 - val_w[...,None]) + rats*val_w[...,None]

        # // Multiply tonescale by final RGB Ratios
        rgb = rats*ts[...,None]

        rgb *= ds
        rgb = np.clip(rgb, 0, clamp_max)
        return rgb


def open_drt_fun(img,parameters:dict):
    def img_to_list(img):
        return img.reshape(-1, 3).tolist()

    def list_to_img(lista,n,m):
        return np.array(lista).reshape(n, m, 3)

    parameters_val = {}
    for clave, valor in parameters.items():
        parameters_val[clave] = dpg.get_value(valor)

    new_img = []
    for pixel in img_to_list(img):
        new_pixel = transform(pixel[0],pixel[1],pixel[2],**parameters_val)
        new_img.append(new_pixel)
    
    new_img = list_to_img(new_img,img.shape[0],img.shape[1])
    return new_img,

def open_drt_fun_V2(img,parameters:dict):
    
    parameters_val = {}
    for clave, valor in parameters.items():
        parameters_val[clave] = dpg.get_value(valor)

    drt = openDRT_class(img,**parameters_val)
    
    return drt.transform(),

class openDRT(NodeV2):
    Title = "openDRT"
    parameters_dic_ID = {}
    def __init__(self):
        f = open_drt_fun_V2
        inp_list = ["img"]
        out_list = ["img"]
        super().__init__(f, inp_list, out_list, self.Title)
        

    def recollect_inputs_frombackawrd_nodes(self):
        # requerimos de un diccionario con todas las variables  y la imagen rgb
        inp = super().recollect_inputs_frombackawrd_nodes()
        inp.append(self.parameters_dic_ID)
        return inp
    
    def custom_node_info(self):
        super().custom_node_info()
        in_gamut_options = ["i_xyz", "i_ap0", "i_ap1", "i_p3d65", "i_rec2020", "i_rec709", "i_awg3", "i_awg4", "i_rwg",
                    "i_sgamut3", "i_sgamut3cine", "i_bmdwg", "i_egamut", "i_davinciwg"]
        
        self.parameters_dic_ID["in_gamut"] = dpg.add_combo       (callback=self.node_modified, label="in gamut", items=in_gamut_options,default_value="i_xyz")
        self.parameters_dic_ID["Lp"] = dpg.add_slider_float(callback=self.node_modified, label="Lp", default_value=100.0, min_value=100.0, max_value=1000.0, width=150)
        self.parameters_dic_ID["gb"] = dpg.add_slider_float(callback=self.node_modified, label="gb", default_value=0.12, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["c"] = dpg.add_slider_float(callback=self.node_modified, label="c", default_value=1.1, min_value=1.0, max_value=1.3, width=150)
        self.parameters_dic_ID["fl"] = dpg.add_slider_float(callback=self.node_modified, label="fl", default_value=0.01, min_value=0.0, max_value=0.02, width=150)
        self.parameters_dic_ID["rw"] = dpg.add_slider_float(callback=self.node_modified, label="rw", default_value=0.2, min_value=0.05, max_value=0.6, width=150)
        self.parameters_dic_ID["bw"] = dpg.add_slider_float(callback=self.node_modified, label="bw", default_value=0.25, min_value=0.05, max_value=0.6, width=150)
        self.parameters_dic_ID["dch"] = dpg.add_slider_float(callback=self.node_modified, label="dch", default_value=0.5, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_toe"] = dpg.add_slider_float(callback=self.node_modified, label="dch_toe", default_value=0.005, min_value=0.0, max_value=0.01, width=150)
        self.parameters_dic_ID["dch_r"] = dpg.add_slider_float(callback=self.node_modified, label="dch_r", default_value=-0.5, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_g"] = dpg.add_slider_float(callback=self.node_modified, label="dch_g", default_value=0.0, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_b"] = dpg.add_slider_float(callback=self.node_modified, label="dch_b", default_value=0.0, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_c"] = dpg.add_slider_float(callback=self.node_modified, label="dch_c", default_value=-0.34, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_m"] = dpg.add_slider_float(callback=self.node_modified, label="dch_m", default_value=-0.34, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["dch_y"] = dpg.add_slider_float(callback=self.node_modified, label="dch_y", default_value=0.0, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["hs_r"] = dpg.add_slider_float(callback=self.node_modified, label="hs_r", default_value=0.4, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["hs_g"] = dpg.add_slider_float(callback=self.node_modified, label="hs_g", default_value=-0.1, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["hs_b"] = dpg.add_slider_float(callback=self.node_modified, label="hs_b", default_value=-0.55, min_value=-1.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_r"] = dpg.add_slider_float(callback=self.node_modified, label="v_r", default_value=0.8, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_g"] = dpg.add_slider_float(callback=self.node_modified, label="v_g", default_value=0.66, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_b"] = dpg.add_slider_float(callback=self.node_modified, label="v_b", default_value=0.5, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_c"] = dpg.add_slider_float(callback=self.node_modified, label="v_c", default_value=0.66, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_m"] = dpg.add_slider_float(callback=self.node_modified, label="v_m", default_value=0.5, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["v_y"] = dpg.add_slider_float(callback=self.node_modified, label="v_y", default_value=0.33, min_value=0.0, max_value=1.0, width=150)
        self.parameters_dic_ID["display_gamut"] = dpg.add_combo       (callback=self.node_modified, label="display_gamut", items=["Rec709", "P3D65", "Rec2020", "None"], default_value="Rec709", width=150)
        self.parameters_dic_ID["EOTF"] = dpg.add_combo       (callback=self.node_modified, label="EOTF", items=["lin", "srgb", "rec1886", "dci", "pq", "hlg"], default_value="rec1886", width=150)

#####################################################
#####################################################


#####################################################
#####################################################

####
####
from modules import COLOR_TRANSFORM, NODE_WINDOW_MENU
from modules.interaction import register
# REGISTRO DE NODOS
with dpg.menu(label=COLOR_TRANSFORM, tag=COLOR_TRANSFORM,parent=NODE_WINDOW_MENU):
    pass

register_list = [
    Inverse_Matrix,
    White_Point,
    RGB_add_White_Point,
    RGB_to_XYZ,
    XYZ_to_RGB,
    COLOR_SPACE,
    EOTF_RGB,
    Ocio_Management,
    openDRT,
    ]
for node in register_list:
    register(node, COLOR_TRANSFORM)
####
####
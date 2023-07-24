import PyOpenColorIO as OCIO
import colour
import rawpy
import pyroexr
import colour_demosaicing

# CONSTANTES
DEBUGGER_MENU = "DEBUGGER_MENU"
NODE_WINDOW = "NODE_WINDOW"
NODE_WINDOW_EDITOR = "node editor"
NODE_WINDOW_MENU = "NODE_WINDOW_MENU"
INSPECTOR_WINDOW = "INSPECTOR_WINDOW"
LOADDERS = "LOADDERS"
MATH = "MATH"
PLOTTERS = "PLOTTERS"
COLOR_TRANSFORM = "COLOR_TRANSFORM"
OPERATORS = "OPERATORS"
WINDOW_SIZE = [1920,1080]
OCIO_PATH = "colormanagement/config.ocio"
OCIO_CONFIG = OCIO.Config.CreateFromFile(OCIO_PATH)


__all__ = [
    "color_transform",
    "loadders",
    "debuggers",
    "father_class",
    "operators", 
    "interaction",
    "math",
    "node_v2",
    "operators",
    "plotters",
    "plantilla",
    "color",
           ]



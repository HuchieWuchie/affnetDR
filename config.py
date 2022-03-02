import sys, inspect
import dataset

class DatasetConfig(object):
    """Base config class for affordanace instance segmentation datasets"""
    
    datasetLoader = dataset.InstanceSegmentationDataSet
    NAME = None
    NUM_CLASSES = 1
    NUM_AFFORDANCES = 1

    OBJ_CLASSES = ('__background__')
    AFF_CLASSES = ('__background__')

    # Affordance colors for mask visualization
    C_BACKGROUND = (0, 0, 0)
    AFF_COLORS = [C_BACKGROUND]

    def __init__(self):
        pass
    

class IITAFF(DatasetConfig):

    NAME = "IIT-AFF"
    NUM_CLASSES = 11
    NUM_AFFORDANCES = 10

    OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife',
                                        'cup', 'drill', 'racket', 'spatula', 'bottle')
    AFF_CLASSES = ('__background__', 'contain', 'cut', 'display', 'engine', 'grasp',
                                        'hit', 'pound', 'support', 'w-grasp')
    
    # Affordance colors for mask visualization
    C_BACKGROUND = (0,0,0)
    C_CONTAIN = (0, 0, 205)
    C_CUT = (0, 192, 0)
    C_DISPLAY = (128,64, 128)
    C_ENGINE = (60, 40, 222)
    C_GRASP = (198, 0, 0)
    C_HIT = (192, 128, 128)
    C_POUND = (0, 222, 222)
    C_SUPPORT = (192, 192, 128)
    C_WGRASP = (0, 134, 141)

    AFF_COLORS = [C_BACKGROUND, C_CONTAIN, C_CUT, C_DISPLAY, C_ENGINE,
                C_GRASP, C_HIT, C_POUND, C_SUPPORT, C_WGRASP]

class IITAFFObject(DatasetConfig):

    NAME = "IIT-AFF-object"
    NUM_CLASSES = 2
    NUM_AFFORDANCES = 10

    datasetLoader = dataset.InstanceSegmentationDataSetObjectness

    OBJ_CLASSES = ('__background__', 'object')
    AFF_CLASSES = ('__background__', 'contain', 'cut', 'display', 'engine', 'grasp',
                                        'hit', 'pound', 'support', 'w-grasp')
    
    # Affordance colors for mask visualization
    C_BACKGROUND = (0,0,0)
    C_CONTAIN = (0, 0, 205)
    C_CUT = (0, 192, 0)
    C_DISPLAY = (128,64, 128)
    C_ENGINE = (60, 40, 222)
    C_GRASP = (198, 0, 0)
    C_HIT = (192, 128, 128)
    C_POUND = (0, 222, 222)
    C_SUPPORT = (192, 192, 128)
    C_WGRASP = (0, 134, 141)

    AFF_COLORS = [C_BACKGROUND, C_CONTAIN, C_CUT, C_DISPLAY, C_ENGINE,
                C_GRASP, C_HIT, C_POUND, C_SUPPORT, C_WGRASP]

class UMD(DatasetConfig):

    NAME = "UMD"
    NUM_CLASSES = 18
    NUM_AFFORDANCES = 8

    datasetLoader = dataset.UMDCategoryDataSet

    OBJ_CLASSES = ('__background__', 'knife', 'saw', 'scissors', 'shears', 'scoop',
                                        'spoon', 'trowel', 'bowl', 'cup', 'ladle',
                                        'mug', 'pot', 'shovel', 'turner', 'hammer',
                                        'mallet', 'tenderizer')
    AFF_CLASSES = ('__background__', 'grasp', 'cut', 'scoop', 'contain', 'pound',
                    'support', 'wrap-grasp')
    
    # Affordance colors for mask visualization
    C_BACKGROUND = (0,0,0)
    C_CONTAIN = (0, 0, 205)
    C_CUT = (0, 192, 0)
    C_SCOOP = (128,64, 128)
    C_GRASP = (198, 0, 0)
    C_POUND = (0, 222, 222)
    C_SUPPORT = (192, 192, 128)
    C_WRAPGRASP = (0, 134, 141)

    AFF_COLORS = [C_BACKGROUND, C_GRASP, C_CUT, C_SCOOP, C_CONTAIN,
                C_POUND, C_SUPPORT, C_WRAPGRASP]


class AFFSynth(DatasetConfig):  

    NAME = "AFF-Synth"
    NUM_CLASSES = 23
    NUM_AFFORDANCES = 11

    datasetLoader = dataset.AFFSynthDataSet

    OBJ_CLASSES = ('__background__', 'knife', 'saw', 'scissors', 'shears', 'scoop',
                                        'spoon', 'trowel', 'bowl', 'cup', 'ladle',
                                        'mug', 'pot', 'shovel', 'turner', 'hammer',
                                        'mallet', 'tenderizer', 'bottle', 'drill', 'monitor', 'pan', 'racket')
    AFF_CLASSES = ('__background__', 'grasp', 'cut', 'scoop', 'contain', 'pound',
                    'support', 'wrap-grasp', 'display', 'engine', 'hit')
    
    # Affordance colors for mask visualization
    C_BACKGROUND = (0,0,0)
    C_GRASP = (0, 0, 255)
    C_CUT = (0, 255, 0)
    C_SCOOP = (122,255, 122)
    C_CONTAIN = (255, 0, 0)
    C_POUND = (255, 255, 0)
    C_SUPPORT = (255, 255, 255)
    C_WRAPGRASP = (255, 0, 255)
    C_DISPLAY = (122, 122, 122)
    C_ENGINE = (0, 255, 255)
    C_HIT = (70, 70, 70)

    AFF_COLORS = [C_BACKGROUND, C_GRASP, C_CUT, C_SCOOP, C_CONTAIN,
                C_POUND, C_SUPPORT, C_WRAPGRASP, C_DISPLAY, C_ENGINE, C_HIT]

def get_classes():

    objs = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            objs.append(obj)
    return objs

def get_class_names():

    names = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if obj.NAME is not None:
                names.append(obj.NAME)
    return names

#get_class_names()

class AFFSynthCollapsed(DatasetConfig):

    NAME = "AFF-Synth-collapsed"
    NUM_CLASSES = 11
    NUM_AFFORDANCES = 11

    datasetLoader = dataset.AFFSynthDataSetCollapsed

    OBJ_CLASSES = ('__background__', 'grasp', 'cut', 'scoop', 'contain', 'pound',
                                        'support', 'wrap-grasp', 'display', 'engine', 'hit')
    AFF_CLASSES = ('__background__', 'grasp', 'cut', 'scoop', 'contain', 'pound',
                    'support', 'wrap-grasp', 'display', 'engine', 'hit')
    CLASS_TO_DATSET_CLASS = {0: 0,
                          1: 2,
                          2: 2,
                          3: 2,
                          4: 2,
                          5: 3,
                          6: 3,
                          7: 3,
                          8: 4,
                          9: 4,
                          10: 4,
                          11: 4,
                          12: 4,
                          13: 6,
                          14: 6,
                          15: 5,
                          16: 5,
                          17: 5,
                          18: 1,
                          19: 9,
                          20: 8,
                          21: 4,
                          22: 10
                          }

    # Affordance colors for mask visualization
    C_BACKGROUND = (0,0,0)
    C_GRASP = (0, 0, 255)
    C_CUT = (0, 255, 0)
    C_SCOOP = (122,255, 122)
    C_CONTAIN = (255, 0, 0)
    C_POUND = (255, 255, 0)
    C_SUPPORT = (255, 255, 255)
    C_WRAPGRASP = (255, 0, 255)
    C_DISPLAY = (122, 122, 122)
    C_ENGINE = (0, 255, 255)
    C_HIT = (70, 70, 70)

    AFF_COLORS = [C_BACKGROUND, C_GRASP, C_CUT, C_SCOOP, C_CONTAIN,
                C_POUND, C_SUPPORT, C_WRAPGRASP, C_DISPLAY, C_ENGINE, C_HIT]

def get_classes():

    objs = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            objs.append(obj)
    return objs

def get_class_names():

    names = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if obj.NAME is not None:
                names.append(obj.NAME)
    return names
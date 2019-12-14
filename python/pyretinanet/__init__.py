import os
import sys

if sys.platform == "linux" or sys.platform == "linux2":
    os.environ["LD_LIBRARY_PATH"] = '/usr/lib/retinanet/:/usr/local/lib/retinanet/'
    sys.path.append('/usr/lib/retinanet/')
    sys.path.append('/usr/local/lib/retinanet/')


from pyretinanetcpp import RetinaNet

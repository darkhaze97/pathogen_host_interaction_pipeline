import imagej
import cv2
ij = imagej.init()
print(ij.op().help())
print('Hi there')
# img = cv2.imread('../Images/Pathogen/A01f00p00d01.tif')
macro = """
open("C:/Users/timhu/cseprojects/pathogen_host_interaction_pipeline/src/Images/Pathogen/A01f00p00d01.tif");
run("Set Measurements...", "area mean min integrated redirect=None decimal=3");
run("Measure");
"""
ij.py.run_macro(macro)
blobs = ij.py.active_image_plus()
ij.py.show(blobs)

# plugin = 'Mean'
# args = {
#     'block_radius_x': 10,
#     'block_radius_y': 10            
# }
# ij.py.run_plugin(plugin, args)
# imp = ij.py.active_image_plus()
# ij.py.show(imp)
# image = ij.py.to_java(img)
# print('Well done')
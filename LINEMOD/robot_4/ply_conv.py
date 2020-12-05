#import open3d as o3d
#mesh = o3d.io.read_triangle_mesh("model1.ply")
#o3d.io.write_triangle_mesh("test.ply", mesh)
import numpy as np
from plyfile import PlyData, PlyElement
plydata = PlyData.read('registeredScene.ply')
el = PlyData([plydata.elements[0],plydata.elements[1]],text=True)
print(plydata)
#PlyData([el], text=True).write('some_ascii.ply')
test_file = './some_ascii.ply'
with open(test_file,'wb') as f:
    plydata.write(f)

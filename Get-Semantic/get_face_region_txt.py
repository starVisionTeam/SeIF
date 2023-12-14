import  os
from h3ds.mesh import Mesh
import numpy as np
from ObjIO import load_obj_data

point_index = []
cn = 0
cn2 = 0
face_region_model_path = '/media/amax/4C76448F76447C28/SePifu_trainDate/template/region/face_and_ear_region/face_and_ear.obj'
face_region_mesh = load_obj_data(face_region_model_path)
# face_region_mesh = Mesh.load(face_region_model_path)

T_model_path = '/media/amax/4C76448F76447C28/刘旭/SePifu_trainDate/template/template_normalization.obj'
T_model_mesh = load_obj_data(T_model_path)
# T_model_mesh = Mesh.load(T_model_path)

for i in np.arange(0,face_region_mesh['v'].shape[0]):
    mask_sphere = np.where(np.linalg.norm(T_model_mesh['v'] - face_region_mesh['v'][i], axis=-1) <= 0.0001)
    point_index.append(mask_sphere[0][0])
    if mask_sphere[0].shape[0]>1:
        cn = cn+1
    if mask_sphere[0].shape[0]==0:
        cn2 =cn2+1

print(cn)
print(cn2)
print(len(point_index))






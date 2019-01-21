import numpy as np

def readOFF(file):
    file = open(file, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    vertst = np.array(verts).T.tolist()
    S = dict()
    S['X'] = np.array(vertst[0])
    S['Y'] = np.array(vertst[1])
    S['Z'] = np.array(vertst[2])
    S['VERTS'] = np.array(verts)
    S['TRIV'] = np.array(faces)
    S['nv'] = n_verts
    return S

import time
import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import eigsh


def cotangent(p):
    return np.cos(p)/np.sin(p)


def cotLaplacian(S):
    
    T1 = S['TRIV'][:,0]
    T2 = S['TRIV'][:,1]
    T3 = S['TRIV'][:,2]

    V1 = S['VERTS'][T1,:]
    V2 = S['VERTS'][T2,:]
    V3 = S['VERTS'][T3,:]

    L1 = np.linalg.norm(V2-V3, axis=1)
    L2 = np.linalg.norm(V1-V3, axis=1)
    L3 = np.linalg.norm(V1-V2, axis=1)
    L = np.column_stack((L1,L2,L3)) #Edges of each triangle

    Cos1 = (L2**2+L3**2-L1**2)/(2*L2*L3)
    Cos2 = (L1**2+L3**2-L2**2)/(2*L1*L3)
    Cos3 = (L1**2+L2**2-L3**2)/(2*L1*L2)
    Cos = np.column_stack((Cos1,Cos2,Cos3)) #Cosines of opposite edges for each triangle 
    Ang = np.arccos(Cos) #Angles 

    I = np.concatenate((T1,T2,T3))
    J = np.concatenate((T2,T3,T1))
    w = 0.5*cotangent(np.concatenate((Ang[:,2],Ang[:,0],Ang[:,1]))).astype(float) 
    In = np.concatenate((I,J,I,J))
    Jn = np.concatenate((J,I,I,J))
    wn = np.concatenate((-w,-w,w,w))
    W = csr_matrix((wn, (In, Jn)), [S['nv'], S['nv']]) #Sparse Cotangent Weight Matrix
    
    cA = cotangent(Ang)/2 #Half cotangent of all angles
    At = 1/4 * (L[:,[1,2,0]]**2 * cA[:,[1,2,0]] +  L[:,[2,0,1]]**2 * cA[:,[2,0,1]]).astype(float) #Voronoi Area

    N = np.cross(V1-V2, V1-V3)
    Ar = np.linalg.norm(N, axis = 1) #Barycentric Area

    #Use Ar is ever cot is negative instead of At
    locs = cA[:,0]<0
    At[locs,0] = Ar[locs]/4;At[locs,1] = Ar[locs]/8;At[locs,2] = Ar[locs]/8;
    
    locs = cA[:,1]<0
    At[locs,0] = Ar[locs]/8;At[locs,1] = Ar[locs]/4;At[locs,2] = Ar[locs]/8;
    
    locs = cA[:,2]<0
    At[locs,0] = Ar[locs]/8;At[locs,1] = Ar[locs]/8;At[locs,2] = Ar[locs]/4;

    Jn = np.zeros(I.shape[0])
    An = np.concatenate((At[:,0], At[:,1], At[:,2]))
    Area = csr_matrix((An, (I, Jn)), [S['nv'],1]) #Sparse Vector of Area Weights

    In = np.arange(S['nv'])
    A = csr_matrix((np.squeeze(np.array(Area.todense())), (In,In)), [S['nv'],S['nv']]) #Sparse Matrix of Area Weights
    return W, A


def eigs_WA(W,A, numEig):
    eigvals, eigvecs = eigsh(W, numEig, A, 1e-6)
    
    return eigvals, eigvecs 


def S_info(S, numEig):
    W, A = cotLaplacian(S)
    t = time.time()
    eigvals, eigvecs = eigs_WA(W, A,numEig)  
    eigvecs_trans = eigvecs.T * A
    return eigvals, eigvecs, eigvecs_trans


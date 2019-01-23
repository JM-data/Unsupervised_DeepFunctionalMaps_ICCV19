from readOFF import *
from laplace_beltrami import *
import scipy.io as sio
import time

n_vecs = 120
file_name = 'tr_reg_'
off_dir = './off_files/'
mat_dir = './Mat_files/'

t = time.time()
for i in range(100):
	print("Getting info for shape : " + file_name + '%.3d.off' % i)
	t1 = time.time()
	file_off = off_dir + file_name + '%.3d.off' % i
	S = readOFF(file_off)
	evals, evecs, evecs_trans = S_info(S, n_vecs)
	params_to_save = {}
	params_to_save['target_evals'] = evals
	params_to_save['target_evecs'] = evecs
	params_to_save['target_evecs_trans'] = evecs_trans
	sio.savemat(mat_dir + file_name + '%.3d.mat' % i, params_to_save)
print('---Done in %f---' % (time.time()-t))


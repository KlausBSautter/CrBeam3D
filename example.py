from cr_beam_3D import beam_element,system
import numpy as np

elements = []
E,A,I = 210000000000.0,0.01,1e-5
G = E/(2.0*(1+0.29))
dx = 0.1
for i in range(40):
    elements.append(beam_element(E,G,I,I,I,A,np.array([[dx*i,0.0,0.0],[dx*(i+1),0.0,0.0]]),[i,i+1]))

sys = system(elements)
fixed_ids = [0,1,2,3,4,5]
toll_e = 1e-4

dof_load_1 = 245

res = np.zeros(sys.max_dof_id)
disp = np.zeros(sys.max_dof_id)
f_ext = np.zeros(sys.max_dof_id)
res_absolut = 1.0


t = 0.0
t_end = 1.0
dt = 0.004

while t<=t_end:

    f_ext[dof_load_1] = 25000*t*1500
    res_absolut = 1.0

    while res_absolut>=toll_e:
        K,F = sys.apply_dirichlet_condition(fixed_ids)
        res = F-f_ext
        du = -1.0*np.linalg.solve(K,res)
        disp = du+disp
        sys.update_elements(disp)
        res_absolut=np.linalg.norm(res)
        print('total residual: ',res_absolut)


    print('time: ',t)
    print('____')


    t+=dt

sys.plot_system()
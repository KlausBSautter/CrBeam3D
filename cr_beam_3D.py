# from "Co-rotational beam elements in instability problems - Jean-Marc Battini"

import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#coordinates [[x1,y1,z1],[x2,y2,z2]]
class beam_element:
    def __init__(self,E,G,Iy,Iz,It,A,coordinates,node_ids):
        self.E,self.A,self.Iy,self.Iz,self.It,self.G = E,A,Iy,Iz,It,G
        self.ref_coordinates = coordinates
        self.deformation_global = np.zeros(12)
        self.deformation_global_last = np.zeros(12)
        self.eft = self.create_eft(node_ids)
        #self.global_total_rotation_matrix_node_1 = self.co_rotating_rotation_matrix(frame="reference")
        #self.global_total_rotation_matrix_node_2 = self.co_rotating_rotation_matrix(frame="reference")

        self.global_total_rotation_matrix_node_1 = np.eye(3)
        self.global_total_rotation_matrix_node_2 = np.eye(3)


    def create_eft(self,node_ids):
        eft = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        for i in range(6):
            eft[i] = int((node_ids[0]*6)+i)
        for i in range(6):
            eft[i+6] = int((node_ids[1]*6)+i)
        return eft

    def current_coordinates(self):
        current_coordinates = dc(self.ref_coordinates)
        current_coordinates[0][0] += self.deformation_global[0]
        current_coordinates[0][1] += self.deformation_global[1]
        current_coordinates[0][2] += self.deformation_global[2]

        current_coordinates[1][0] += self.deformation_global[6]
        current_coordinates[1][1] += self.deformation_global[7]
        current_coordinates[1][2] += self.deformation_global[8]
        return current_coordinates

    def length(self,coordinates):
        dz = coordinates[1][2]-coordinates[0][2]
        dy = coordinates[1][1]-coordinates[0][1]
        dx = coordinates[1][0]-coordinates[0][0]
        return np.sqrt(np.power(dx,2.0)+np.power(dy,2.0)+np.power(dz,2.0))

    def local_stiffness_matrix(self):
        C = np.zeros((7,7))
        C[0,0] = self.E*self.A

        C[1,1],C[4,4] = self.G*self.It,self.G*self.It
        C[1,4],C[4,1] = -self.G*self.It,-self.G*self.It

        C[2,2],C[5,5] = 4.0 * self.E*self.Iz,4.0 * self.E*self.Iz
        C[2,5],C[5,2] = 2.0 * self.E*self.Iz,2.0 * self.E*self.Iz

        C[3,3],C[6,6] = 4.0 * self.E*self.Iy,4.0 * self.E*self.Iy
        C[3,6],C[6,3] = 2.0 * self.E*self.Iy,2.0 * self.E*self.Iy

        C /= self.length(self.ref_coordinates)
        return C

    def local_stiffness_matrix_a(self):
        K_a = np.zeros((7,7))
        b_a = self.b_matrix_a()
        K_a += (b_a.T)@self.local_stiffness_matrix()@b_a

        K_h = np.zeros((7,7))
        f_int = self.local_internal_forces()
        d_local = self.local_deformations()
        for i,node_nr in enumerate([1,2]):
            T_inv = self.rotation_vector_inverse_T_s(node_id=node_nr)
            if node_nr==1:
                m_i = f_int[1:4]
                phi_i = d_local[1:4]
            elif node_nr==2:
                m_i = f_int[4:7]
                phi_i = d_local[4:7]

            alpha = np.linalg.norm(phi_i)

            k_h_i = np.zeros((3,3))

            if alpha>0.0:
                eta = 2.0*np.sin(alpha) - alpha*(1.0+np.cos(alpha))
                eta /= 2.0*alpha*alpha*np.sin(alpha)

                nu = (alpha*(alpha+np.sin(alpha))) - (8.0*np.power(np.sin(alpha/2.0),2.0))
                nu /= 4.0*np.power(alpha,4.0)*np.power(np.sin(alpha/2.0),2.0)

                phi_tiled = self.create_skew_symmetric_matrix(phi_i)
                m_tiled = self.create_skew_symmetric_matrix(m_i)

                k_h_i = eta * (np.outer(phi_i,m_i) - 2.0*np.outer(m_i,phi_i) + (phi_i@m_i)*np.eye(3))
                k_h_i += nu * (phi_tiled@phi_tiled) * np.outer(m_i,phi_i)
                k_h_i -= 0.50 * m_tiled

                k_h_i = k_h_i@self.rotation_vector_inverse_T_s(node_nr)


            if node_nr==1:
                K_h[1:4,1:4] = k_h_i
            elif node_nr==2:
                K_h[4:7,4:7] = k_h_i



        K_a += K_h
        return K_a

    def global_tangent_stiffness_matrix(self):
        b_g = self.b_matrix_g()
        K_g = (b_g.T)@self.local_stiffness_matrix_a()@b_g


        m_plus_1 = self.local_internal_forces_a()
        m = m_plus_1[1:7]


        Q_vec = (self.P_matrix().T)@m
        Q = np.zeros((12,3))
        Q[0:3,0:3] = self.create_skew_symmetric_matrix(Q_vec[0:3])
        Q[3:6,0:3] = self.create_skew_symmetric_matrix(Q_vec[3:6])
        Q[6:9,0:3] = self.create_skew_symmetric_matrix(Q_vec[6:9])
        Q[9:12,0:3] = self.create_skew_symmetric_matrix(Q_vec[9:12])

        E = self.E_matrix()
        G = self.G_matrix()

        l_n = self.length(self.current_coordinates())
        eta = G[2,0]*l_n

        a = np.array([[0.0,((eta/l_n) * (m[0]+m[3])) - ((1.0/l_n) * (m[1]+m[4])),(m[2]+m[5])/l_n]]).T


        r_vec_mat = np.zeros((1,12))
        r_vec_mat[0,0:12] = self.r_vector()


        K_m = m_plus_1[0]*self.D_matrix()
        K_m -= E@Q@(G.T)@(E.T)
        K_m += E@G@a@r_vec_mat

        K_g = K_g+K_m

        return K_g

    def create_skew_symmetric_matrix(self,vec_i):
        sm = np.zeros((3,3))
        sm[0,1] = -vec_i[2]
        sm[0,2] = vec_i[1]
        sm[1,0] = vec_i[2]
        sm[1,2] = -vec_i[0]
        sm[2,0] = -vec_i[1]
        sm[2,1] = vec_i[0]
        return sm

    def local_internal_forces(self):
        local_deformations = self.local_deformations()
        local_stiffness_matrix = self.local_stiffness_matrix()
        return local_stiffness_matrix@local_deformations

    def local_internal_forces_a(self):
        return (self.b_matrix_a().T)@self.local_internal_forces()

    def global_internal_forces(self):
        return (self.b_matrix_g().T)@self.local_internal_forces_a()

    def local_deformations(self):
        disp_l = np.zeros(7)
        disp_l[0] = self.length(self.current_coordinates()) - self.length(self.ref_coordinates)

        R_co_rotating_reference = self.co_rotating_rotation_matrix(frame="reference")
        R_co_rotating_current = self.co_rotating_rotation_matrix(frame="current")

        R_local_node_1 = (R_co_rotating_current.T)@self.global_total_rotation_matrix_node_1@R_co_rotating_reference
        R_local_node_2 = (R_co_rotating_current.T)@self.global_total_rotation_matrix_node_2@R_co_rotating_reference


        local_rot_norm_node_1 = np.arccos((np.trace(R_local_node_1)-1.0)/2.0)
        local_rot_norm_node_2 = np.arccos((np.trace(R_local_node_2)-1.0)/2.0)

        if local_rot_norm_node_1!=0.00:
            log_R_local_node_1 = (local_rot_norm_node_1/(2.0*np.sin(local_rot_norm_node_1)))*(R_local_node_1-(R_local_node_1.T))
            disp_l[1],disp_l[2],disp_l[3] = log_R_local_node_1[2,1],log_R_local_node_1[0,2],log_R_local_node_1[1,0]

        if local_rot_norm_node_2!=0.00:
            log_R_local_node_2 = (local_rot_norm_node_2/(2.0*np.sin(local_rot_norm_node_2)))*(R_local_node_2-(R_local_node_2.T))
            disp_l[4],disp_l[5],disp_l[6] = log_R_local_node_2[2,1],log_R_local_node_2[0,2],log_R_local_node_2[1,0]

        return disp_l

    def update_global_total_rotation_matrix(self):
        d_disp = self.increment_deformation_global()
        d_phi_node_1 = d_disp[3:6]
        d_phi_node_2 = d_disp[9:12]

        ## update node 1
        d_phi_tilde = np.array([[0.0,-d_phi_node_1[2],d_phi_node_1[1]],
                                [d_phi_node_1[2],0.0,-d_phi_node_1[0]],
                                [-d_phi_node_1[1],d_phi_node_1[0],0.0]])
        d_phi_norm = np.linalg.norm(d_phi_node_1)

        if d_phi_norm>0.0:
            R_update  = np.eye(3)
            R_update += (np.sin(d_phi_norm)/d_phi_norm) * d_phi_tilde
            R_update += 0.5 * np.power(np.sin(d_phi_norm/2.0)/(d_phi_norm/2.0),2.0) * (d_phi_tilde@d_phi_tilde)

            self.global_total_rotation_matrix_node_1 = R_update@self.global_total_rotation_matrix_node_1

        ## update node 2
        d_phi_tilde = np.array([[0.0,-d_phi_node_2[2],d_phi_node_2[1]],
                                [d_phi_node_2[2],0.0,-d_phi_node_2[0]],
                                [-d_phi_node_2[1],d_phi_node_2[0],0.0]])
        d_phi_norm = np.linalg.norm(d_phi_node_2)

        if d_phi_norm>0.0:
            R_update  = np.eye(3)
            R_update += (np.sin(d_phi_norm)/d_phi_norm) * d_phi_tilde
            R_update += 0.5 * np.power(np.sin(d_phi_norm/2.0)/(d_phi_norm/2.0),2.0) * (d_phi_tilde@d_phi_tilde)

            self.global_total_rotation_matrix_node_2 = R_update@self.global_total_rotation_matrix_node_2

    def co_rotating_rotation_matrix(self,frame):
        R = np.zeros((3,3))
        if frame=="current":
            current_coordinates = self.current_coordinates()
            reference_rotation = self.co_rotating_rotation_matrix(frame="reference")

            r_1 = current_coordinates[1]-current_coordinates[0]
            r_1 /= np.linalg.norm(r_1)

            q_1 = self.global_total_rotation_matrix_node_1@reference_rotation@(np.array([[0.0,1.0,0.0]]).T)
            q_2 = self.global_total_rotation_matrix_node_2@reference_rotation@(np.array([[0.0,1.0,0.0]]).T)

            q_mean = 0.5 * (q_1+q_2)

            r_3 = np.cross(r_1,q_mean[:,0])
            r_3 /= np.linalg.norm(r_3)

            r_2 = np.cross(r_3,r_1)
            r_2 /= np.linalg.norm(r_2)

            R[:,0],R[:,1],R[:,2] = r_1,r_2,r_3
            return R


        elif frame=="reference":
            r_1 = self.ref_coordinates[1]-self.ref_coordinates[0]
            r_1 /= np.linalg.norm(r_1)

            #this might cause problems
            r_2 = np.cross(r_1,np.array([0.0,0.0,1.0]))
            r_2 /= np.linalg.norm(r_2)

            r_3 = np.cross(r_1,r_2)
            r_3 /= np.linalg.norm(r_3)

            R[:,0],R[:,1],R[:,2] = r_1,r_2,r_3

            return R

        else:
            print('frame: ',frame,' not known')
            quit()

    def increment_deformation_global(self):
        return self.deformation_global-self.deformation_global_last

    def rotation_vector_inverse_T_s(self,node_id):
        if node_id==1:
            disp_l = self.local_deformations()[1:4]
        elif node_id==2:
            disp_l = self.local_deformations()[4:7]
        else:
            print('node nr: ',node_id,' not known')
            quit()

        phi_norm = np.linalg.norm(disp_l)
        T_inv = np.eye(3)
        if phi_norm>0.0:
            u_vec = disp_l/phi_norm
            phi_tilde = np.array([[0.0,-disp_l[2],disp_l[1]],
                                [disp_l[2],0.0,-disp_l[0]],
                                [-disp_l[1],disp_l[0],0.0]])

            T_inv  = ( (phi_norm/2.0) / (np.tan(phi_norm/2.0)) )*np.eye(3)
            T_inv += (1.0 - ((phi_norm/2.0)/(np.tan(phi_norm/2.0)))) * np.outer(u_vec,u_vec)
            T_inv -= 0.5 * phi_tilde

        return T_inv

    def b_matrix_a(self):
        b = np.zeros((7,7))
        b[0,0] = 1.0
        b[1:4,1:4] = self.rotation_vector_inverse_T_s(node_id=1)
        b[4:7,4:7] = self.rotation_vector_inverse_T_s(node_id=2)
        return b

    def finalize_element(self):
        ## maybe change this...
        self.update_global_total_rotation_matrix()
        self.deformation_global_last = dc(self.deformation_global)

    def r_vector(self):
        current_coordinates = self.current_coordinates()
        r_1 = current_coordinates[1]-current_coordinates[0]
        r_1 /= np.linalg.norm(r_1)

        r_vec = np.zeros(12)
        r_vec[0:3] = -r_1
        r_vec[6:9] = r_1
        return r_vec

    def E_matrix(self):
        R_r = self.co_rotating_rotation_matrix(frame="current")
        E = np.zeros((12,12))
        E[0:3,0:3] = R_r
        E[3:6,3:6] = R_r
        E[6:9,6:9] = R_r
        E[9:12,9:12] = R_r
        return E

    def G_matrix(self):
        reference_rotation = self.co_rotating_rotation_matrix(frame="reference")
        current_rotation = self.co_rotating_rotation_matrix(frame="current")

        q_1 = self.global_total_rotation_matrix_node_1@reference_rotation@(np.array([[0.0,1.0,0.0]]).T)
        q_2 = self.global_total_rotation_matrix_node_2@reference_rotation@(np.array([[0.0,1.0,0.0]]).T)

        q_mean = 0.5 * (q_1+q_2)

        q_i = ((current_rotation.T)@q_mean)[:,0]
        q_1i = ((current_rotation.T)@q_1)[:,0]
        q_2i = ((current_rotation.T)@q_1)[:,0]

        eta = q_i[0]/q_i[1]
        eta_11 = q_1i[0]/q_i[1]
        eta_12 = q_1i[1]/q_i[1]
        eta_21 = q_2i[0]/q_i[1]
        eta_22 = q_2i[1]/q_i[1]

        l_n = self.length(self.current_coordinates())

        G_transposed = np.zeros((3,12))
        G_transposed[0,2] = eta/l_n
        G_transposed[0,3] = eta_12/2.0
        G_transposed[0,4] = -eta_11/2.0
        G_transposed[0,8] = -eta/l_n
        G_transposed[0,9] = eta_22/2.0
        G_transposed[0,10] = -eta_21/2.0

        G_transposed[1,2] = 1.0/l_n
        G_transposed[1,8] = -1.0/l_n

        G_transposed[2,1] = -1.0/l_n
        G_transposed[2,7] = 1.0/l_n

        return G_transposed.T

    def P_matrix(self):
        P = np.zeros((6,12))
        P[0:3,3:6] = np.eye(3)
        P[3:6,9:12] = np.eye(3)


        G_transposed = self.G_matrix().T
        p_temp = np.zeros((6,12))
        p_temp[0:3,0:12] = G_transposed
        p_temp[3:6,0:12] = G_transposed

        P = P - p_temp
        return P

    def b_matrix_g(self):
        r_vec = self.r_vector()
        P_E_t = self.P_matrix()@(self.E_matrix().T)

        b_mat_g = np.zeros((7,12))
        b_mat_g[0,0:12] = r_vec
        b_mat_g[1:7,0:12] = P_E_t

        return b_mat_g

    def D_matrix(self):
        current_coordinates = self.current_coordinates()
        r_1 = current_coordinates[1]-current_coordinates[0]
        r_1 /= np.linalg.norm(r_1)

        l_n = self.length(self.current_coordinates())

        D_3 = (np.eye(3) - np.outer(r_1,r_1))/l_n

        D = np.zeros((12,12))
        D[0:3,0:3] = D_3
        D[0:3,6:9] = -D_3
        D[6:9,0:3] = -D_3
        D[6:9,6:9] = D_3

        return D

class system:
    def __init__(self,elements):
        self.elements = elements
        self.max_dof_id = self.max_dof()

    def max_dof(self):
        max_dof = 0
        for element in self.elements:
            for id in element.eft:
                if id>max_dof: max_dof=id
        return max_dof+1

    def stiffness_matrix(self):
        K = np.zeros((self.max_dof_id,self.max_dof_id))
        for element in self.elements:
            k_ele = element.global_tangent_stiffness_matrix()
            eft = element.eft

            for i,eft_i in enumerate(eft):
                for j,eft_j in enumerate(eft):
                    K[eft_i,eft_j] += k_ele[i,j]
        return K

    def internal_forces(self):
        F = np.zeros(self.max_dof_id)
        for element in self.elements:
            f_ele = element.global_internal_forces()
            eft = element.eft

            for i,eft_i in enumerate(eft):
                F[eft_i] += f_ele[i]
        return F

    def apply_dirichlet_condition(self,fixed_ids):
        K = self.stiffness_matrix()
        F = self.internal_forces()

        for id in fixed_ids:
            for i in range(self.max_dof_id):
                K[i,id] = 0.0
                K[id,i] = 0.0

        for id in fixed_ids:
            F[id]=0.0
            K[id,id] = 1.0

        return K,F

    def update_elements(self,displacement):
        for element in self.elements:
            eft = element.eft

            for i,eft_i in enumerate(eft):
                element.deformation_global[i] = displacement[eft_i]

            element.finalize_element()

    def plot_system(self):
        x,y,z = [],[],[]
        for element in self.elements:
            current_coordinates = element.current_coordinates()
            x.append(current_coordinates[0][0])
            x.append(current_coordinates[1][0])
            y.append(current_coordinates[0][1])
            y.append(current_coordinates[1][1])
            z.append(current_coordinates[0][2])
            z.append(current_coordinates[1][2])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x,y,z)
        plt.show()

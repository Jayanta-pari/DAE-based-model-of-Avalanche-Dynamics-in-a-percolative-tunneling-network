
############################################################################################

# prompt: can you make edges prominant , cant see clearly

import networkx as nx
import numpy as np
import networkx as nx
import pandas as pd
import math
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import svds
import random
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize,Bounds,nnls,least_squares
from scipy.linalg import null_space,lstsq
# import sympy as sp
from scipy.sparse.linalg import cg,gmres,spsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import scipy.linalg as la
import h5py
from concurrent.futures import ThreadPoolExecutor
# from numba import njit, prange
# from dask import delayed, compute
# import ray
import threading
from concurrent.futures import ProcessPoolExecutor
import time
import os
import random
import math
import matplotlib.pyplot as plt
rf=[8/9]
icut=[0.0001]
for RF in rf:
    for I_cut in icut:
        starttime=time.time()
        def create_3d_layered_graph(box_size, num_layers, points_per_layer, z_plate, d, max_d):
            G = nx.DiGraph()
            layer_height = box_size / num_layers
            existing_nodes = set()
            
            # Generate nodes for each layer in the plate (only node creation, no intra-plate edges)
            for layer in range(num_layers):
                y_offset = box_size - (layer + 1) * layer_height
                for _ in range(points_per_layer):
                    while True:
                        x = random.uniform(0, box_size)
                        y = random.uniform(y_offset, y_offset + layer_height)
                        new_node = (x, y, z_plate)
                        # Ensure minimum 2D separation on the same z-plane
                        if all(math.hypot(nx - x, ny - y) > d for (nx, ny, nz) in existing_nodes if nz == z_plate):
                            G.add_node(new_node, layer=layer)
                            existing_nodes.add(new_node)
                            break
            return G

        def create_3d_cube_graph(box_size, num_plates, num_layers, points_per_layer, d, max_d, inter_d, inter_max_d):
            global_graph = nx.DiGraph()
            z_positions = np.linspace(0, box_size, num_plates)
            
            # Create plates (nodes only, no intra-plate edges)
            for z in z_positions:
                plate_g = create_3d_layered_graph(box_size, num_layers, points_per_layer, z, d, max_d)
                global_graph = nx.compose(global_graph, plate_g)
            
            # Create global input and output nodes
            layer_height = box_size / num_layers
            global_input_node = (box_size/2, box_size + layer_height, z_positions[0])
            global_output_node = (box_size/2, -layer_height, z_positions[-1])
            global_graph.add_node(global_input_node, layer=-1)
            global_graph.add_node(global_output_node, layer=num_layers)
            
            # Connect global input to all nodes in the first plate
            first_plate_z = z_positions[0]
            first_plate_nodes = [n for n in global_graph.nodes if n[2] == first_plate_z]
            for node in first_plate_nodes:
                dx = global_input_node[0] - node[0]
                dy = global_input_node[1] - node[1]
                distance = math.hypot(dx, dy)
                global_graph.add_edge(global_input_node, node, distance=distance)
            
            # Connect all nodes in the last plate to global output
            last_plate_z = z_positions[-1]
            last_plate_nodes = [n for n in global_graph.nodes if n[2] == last_plate_z]
            for node in last_plate_nodes:
                dx = node[0] - global_output_node[0]
                dy = node[1] - global_output_node[1]
                distance = math.hypot(dx, dy)
                global_graph.add_edge(node, global_output_node, distance=distance)
            
            # Enhanced inter-plate (vertical) connections: connect each node to the nearest node in the next plate
            for i in range(num_plates - 1):
                current_z = z_positions[i]
                next_z = z_positions[i + 1]
                
                current_plate_nodes = [n for n in global_graph.nodes if n[2] == current_z]
                next_plate_nodes = [n for n in global_graph.nodes if n[2] == next_z]
                
                # for src in current_plate_nodes:
                #     min_distance = float('inf')
                #     closest_node = None
                #     for dst in next_plate_nodes:
                #         dx = src[0] - dst[0]
                #         dy = src[1] - dst[1]
                #         dz = src[2] - dst[2]
                #         distance = math.sqrt(dx**2 + dy**2 + dz**2)
                #         if inter_d <= distance <= inter_max_d and distance < min_distance:
                #             min_distance = distance
                #             closest_node = dst
                #     if closest_node:
                #         global_graph.add_edge(src, closest_node, distance=min_distance)

                ##only hrizontal distance###

                for src in current_plate_nodes:
                    min_horizontal_dist = float('inf')
                    closest_node = None
                    for dst in next_plate_nodes:
                        dx = src[0] - dst[0]
                        dy = src[1] - dst[1]
                        horizontal_dist = math.hypot(dx, dy)
                        if inter_d <= horizontal_dist <= inter_max_d and horizontal_dist < min_horizontal_dist:
                            min_horizontal_dist = horizontal_dist
                            closest_node = dst
                    if closest_node:
                        dz = closest_node[2] - src[2]
                        full_3d_distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        global_graph.add_edge(src, closest_node, distance=full_3d_distance)

                        
            pos = {node: node for node in global_graph.nodes()}
            print(f"position is:{pos}")
            global_graph = nx.convert_node_labels_to_integers(global_graph, label_attribute='pos')
            return global_graph, global_input_node, global_output_node, pos

        # Parameters
        box_size = 10
        num_plates = 5
        num_layers = 12  # Reduced for better visualization
        points_per_layer = 30
        d = 0.35
        max_d = 0.71
        # inter_d = 2.5
        inter_d=0
        # inter_max_d = 3.5
        inter_max_d=2

        # Create graph
        G, global_input_node, global_output_node, pos = create_3d_cube_graph(
            box_size, num_plates, num_layers, points_per_layer, d, max_d, inter_d, inter_max_d
        )
        input_node_id = [n for n in G.nodes if np.allclose(G.nodes[n]['pos'], global_input_node)][0]
        output_node_id = [n for n in G.nodes if np.allclose(G.nodes[n]['pos'], global_output_node)][0]

        # Visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        pos_dict = nx.get_node_attributes(G, 'pos')
        pos_list = list(pos_dict.values())
        xs, ys, zs = zip(*pos_list)

        # Color coding
        colors = []
        for node in G.nodes():
            if node == input_node_id:
                colors.append('red')
            elif node == output_node_id:
                colors.append('blue')
            else:
                z_val = pos_dict[node][2]
                colors.append(plt.cm.viridis(z_val/box_size))

        colors = np.array(colors, dtype=object)

        # Plot nodes
        ax.scatter(xs, ys, zs, c=colors, s=30, alpha=0.7)

        # Plot edges
        for edge in G.edges():
            u, v = edge
            start = pos_dict[u]
            end = pos_dict[v]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color='gray', alpha=0.5, linewidth=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("3D Cube Graph with Only Vertical (Inter-Plate) Connections")
        plt.show()

        # input_node = None
        # output_node=None
        # for node, data in G.nodes(data=True):
        #     if data.get('layer') == -1:  # input node has layer -1
        #         input_node = node
        #     if data.get('layer') == num_layers:  # output node has layer num_layers
        #         output_node = node
        #         break
        # print(f"output data is:{output_node}")
        # print(f"input data is:{input_node}")

        output_degree = G.degree[output_node_id]
        print(f"The number of edges connected to the output node: {output_degree}")
        edge_list = [(u, v,data) for u, v, data in G.edges(data=True)]
        # print(edge_list)
        num_nodes = G.number_of_nodes()
        num_edges = len(edge_list)

        def compute_matrices(G,edge_list,input_node_id,output_node_id):
            # edges = list(G.edges(data=True))
            num_nodes = G.number_of_nodes()
            num_edges = len(edge_list)
            G0=1
            beta=100
            # Initialize matrices
            M = np.zeros((num_nodes, num_edges))  # Incidence matrix
            D = np.zeros((num_edges, num_edges))  # Distance matrix
            G_matrix = np.zeros((num_edges, num_edges)) 
            node_index = {node: i for i, node in enumerate(G.nodes())}  # Create a mapping

            for edge_idx, (u, v, data) in enumerate(edge_list):
                M[node_index[u], edge_idx] = -1   
                M[node_index[v], edge_idx] = 1  
                D[edge_idx, edge_idx] = data['distance']
                if u not in (input_node_id,output_node_id) and v not in (input_node_id,output_node_id):
                    G[u][v]['flag']=False
                    G[u][v]['Y']=G0*np.exp(-beta*D[edge_idx, edge_idx])
                    G_matrix[edge_idx, edge_idx]= G[u][v]['Y']
                    G[u][v]['z']=0
                else:
                    G[u][v]['flag']=True
                    G[u][v]['Y']=0.933
                    G_matrix[edge_idx, edge_idx]= G[u][v]['Y']
                    G[u][v]['z']=D[edge_idx, edge_idx]
            
            return M, D,G_matrix,G

        # Generate graph and compute matrices
        # box_size = 10
        # num_layers = 9
        # points_per_layer = 50
        # G = create_layered_graph(box_size, num_layers, points_per_layer)
        M, D,G_matrix,G = compute_matrices(G,edge_list,input_node_id,output_node_id)

        # Visualization
        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        # nx.draw(G, pos, node_size=5, arrowsize=8, with_labels=False)
        # plt.title('Directed Graph with Integer Nodes')

        # plt.subplot(122)
        # plt.spy(D)
        # plt.title('Distance Matrix (D)')
        # plt.show()

        print("Incidence Matrix Shape:", M.shape)
        print("Distance Matrix Shape:", D.shape)
        print(f"maximum of D matrix is :{np.max(np.diag(D))}")
        print(f"minimum of D is :{np.min(np.diag(D))}")

        # input_node = None
        # output_node=None
        # for node, data in G.nodes(data=True):
        #     if data.get('layer') == -1:  # input node has layer -1
        #         input_node = node
        #     if data.get('layer') == num_layers:  # output node has layer num_layers
        #         output_node = node
        #         break
        print(f"output data is:{input_node_id}")
        input_degree = G.degree[input_node_id]
        print(f"The number of edges connected to the output node: {input_degree}")

        beta = 100
        kappa = 0.038
        #kappa=0.018
        mu = 0.346
        #mu=0.412
        G0=1
        delta_t = 0.01  

        L = np.zeros((num_edges, num_edges))  
        # D = np.zeros((num_edges, num_edges))  


        # edge_list = {edge: idx for idx, edge in enumerate(G.edges())}
        node_indices = {node: idx for idx, node in enumerate(G.nodes())}



        print(f"M matrix is:{M}")
        #L = D.copy() 
        col_sums = np.sum(M, axis=0)
        print(f"coloumn sum is :{col_sums}")

        static_edges = set()
        for idx,(u, v,data) in enumerate(edge_list): 
            # if input_node in (u, v) or output_node in (u, v): 
            if node_indices[u] in (input_node_id,output_node_id) or node_indices[v] in (input_node_id,output_node_id):

                static_edges.add((u, v)) 
        #         G[u][v]['Y']=1
        #         G_matrix[idx, idx]= G[u][v]['Y']
                #G[u][v]['flag']=True
                #static_edges.add((v, u)) 
        t = 0.0
        T = 100


        # G_matrix = G0*(np.diag(np.exp(-beta * np.diag(L))))
        # print(f"G_matrix is :{G_matrix}")

        # print(M.shape)


        # bounds = Bounds(0, np.inf)

        power=[]
        current=[]
        voltage=[]
        conductance=[]

        time1=np.arange(0,T,delta_t)
        V_initial=np.zeros(num_nodes)
        # V_internal=np.zeros(num_nodes-2)



        def kaczmarz_method(M,G_matrix,V_input, V_output,V_initial,b, tol=1e-6, max_iter=1000,lr=1e-3):
        # def kaczmarz_method(G, Vin, sourcenode, groundnode):
            
            A=np.matmul(M,np.matmul(G_matrix,M.T))
            epsilon=1e-10
            A = A + epsilon * np.eye(A.shape[0])
            A=A/np.linalg.norm(A)
            m, n = A.shape
            print(f"rank of matrix is:{np.linalg.matrix_rank(A)}")


            V_max=V_input
            V_min=V_output
            X = cp.Variable(n)
            

            constraints = [
            
            X[0] == V_input,# Boundary condition at X[0]

            X[-1] == V_output,  # Boundary condition at X[-1]

            X<=V_max,
            X>=V_min
            ]
            # objective=cp.Minimize(cp.norm(A @ X,2))
            objective=cp.Minimize(cp.sum_squares(A@X))
            # objective=cp.Minimize(A@X)
            problem = cp.Problem(objective, constraints)

            
            problem.solve(verbose=True,solver='ECOS')



            return X.value


            
            
        def update_filament_growth(idx,u,v,G, D_idx,dzdotbydz,V_jn,Lij,G_id, V_set=0.01, V_reset=0.005, L_max=np.max(D), beta=50, mu=0.346,kappa=0.038, delta_t=0.01,RF=RF,I_cut=I_cut):
            G_cut=0.8
            G_joule=0.01
            epsilon = 1e-6 
            # if D_idx>0:
            max_d=0.69
            I_cut=I_cut
            G_cut=0.001
            Gon=0.93
            RF=RF
            currentfactor=0.1
            IT=(I_cut*currentfactor)
            # IT=0.989
            # if G_id<=G_joule:
            # if Lij>=0.35 and Lij<max_d:
            # if (D_idx*(1-Z_old))>=0.35 and (D_idx*(1-Z_old))<(0.99*max_d):
            # if (D_idx*(1-Z_old))>=0 and (D_idx*(1-Z_old))<(0.99*max_d):
            
            # if Z_old>=0 and Z_old<0.999:
            if not G[u][v]['flag']:

                G[u][v]['z'] = ((1-kappa*delta_t)*(G[u][v]['z']) + kappa*delta_t*((((mu*V_jn)/(kappa*D_idx**2))/(1-(G[u][v]['z'])))))

                if G[u][v]['z']>=0.999:
                    # Z_new=0.999
                    G[u][v]['z']=0.999
                    G[u][v]['Y']=Gon
                    G_id=G[u][v]['Y']
                    I_new=V_jn*G_id
                    # dzdotbydz=0
                    dzdotbydz=1-((4*mu*V_jn)/(kappa*D_idx**2))
                    # I_new=I_cut
                    if G_id>=I_cut:
                    #if G_id>=(0.01*Gon):
                        G[u][v]['flag']=True
                    #     # Z_old=D_idx-Lij
                    #     Z_old=0.999
                else:
                    # G[u][v]['Y']=G0*np.exp(-beta*(D_idx-(G[u][v]['z'])))
                    G[u][v]['Y']=G0*np.exp(-beta*D_idx*(1-(G[u][v]['z'])))
                    G_id=G[u][v]['Y']
                    I_new=V_jn*G_id
                    # if G_id>=(G_cut):
                    #    G[u][v]['flag']=True
                    #dzdotbydz=((((mu*V_jn)/((D_idx**2)*(1-G[u][v]['z'])**2))) -kappa)
                    dzdotbydz=1-((4*mu*V_jn)/(kappa*D_idx**2))

                # return G_id,I_new,dzdotbydz
            else:
                # print("hi i am decreasing")
                # Z_old= max(0,Z_old-(1/RF))
                G[u][v]['z']=G[u][v]['z']*RF
                # Z_old=Z_old/D_idx
                # G_id=G0*np.exp(-beta*D_idx*(1-Z_old))
                G[u][v]['Y']=G0*np.exp(-beta*D_idx*(1-(G[u][v]['z'])))
                # G[u][v]['Y']=G0*np.exp(-beta*(D_idx-(G[u][v]['z'])))
                G_id=G[u][v]['Y']
                I_new=V_jn*G_id
                dzdotbydz=1-((4*mu*V_jn)/(kappa*D_idx**2))
                if G_id<= abs(I_cut-IT):

                    G[u][v]['flag']=False



            return G_id,I_new,dzdotbydz,G[u][v]['z']




        def parallel_filament_update(idx, u, v,data,V,D,dzdt,input_node_id,output_node_id, L,G_matrix, L_min, delta_t,static_edges,G,RF=RF,I_cut=I_cut):
            

            # if u in (input_node,output_node) or v in (input_node,output_node):
            # if u == input_node or v == input_node or u == output_node or v == output_node:
            # continue  # Skip input/output edges

                    
                # return idx, Z_old[idx, idx], 1,0,0

            

            # if node_indices[u] not in (input_node,output_node) and node_indices[v] not in (input_node,output_node):
            if u not in (input_node_id,output_node_id) and v not in (input_node_id,output_node_id):
                V_jn = abs(V[node_indices[u]] - V[node_indices[v]])

                G_id,I_new,dzdotbydz,Z_new = update_filament_growth(idx,u,v,G,D[idx,idx],dzdt[idx,idx],V_jn,L[idx,idx],G_matrix[idx,idx], L_max=np.max(D),RF=RF,I_cut=I_cut)

                # I_new=G_matrix[idx,idx]*(V_jn)

            

                return idx,G_id,I_new,dzdotbydz,Z_new

            else:
                return idx, 0.93,0,0,G[u][v]['z']
            



        # Update voltage at each time step
        V_input = 1
        V_output = 0
        V_internal_initial = np.zeros(num_nodes - 2)
        Z_old=np.zeros((num_edges,num_edges))

        I=np.zeros(num_edges)
        L_thresold=1
        b=np.zeros(num_nodes)
        V_initial[0]=V_input
        V_initial[-1]=V_output
        L_new=np.zeros((num_edges,num_edges))
        V_set=0.01
        V_reset=0.005
        # time=[]
        L_min=1e-3
        timestamp=[]




        eigenvalues_list = []
        mean_eigenvalue_list=[]
        min_eigenvalue_list=[]

        # Small epsilon for safe division
        epsilon = 1e-6

        # Sparse matrix format for Z and dZ/dt
        Z_storage_sparse1 = []
        dZ_dt_storage_sparse1 = []
        Z_storage_sparse2 = []
        dZ_dt_storage_sparse2 = []
        dz2dt2_storage_sparse=[]

        dzdt=np.zeros((num_edges,num_edges))

        dz2dt2=np.zeros((num_edges,num_edges))
        Z_sparse = csr_matrix(Z_old)
        dZ_dt_sparse = csr_matrix(dzdt)
        dz2dt2_sparce=csr_matrix(dz2dt2)


        dz2dt2_storage_sparse.append(dz2dt2_sparce)
        Z_storage_sparse2.append(Z_sparse)
        dZ_dt_storage_sparse2.append(dZ_dt_sparse)
        Identity=np.eye(num_edges)
        L=D*(Identity-Z_old)
        print(f"L matrix is:{L}")
        # G_matrix = G0 * (np.diag(np.exp(-beta * np.diag(L))))
        G_undirected = G.to_undirected()
        Rnetwork_list= nx.resistance_distance(G_undirected,input_node_id,output_node_id, weight='Y', invert_weight=False)   
        Ynetwork_list = 1/Rnetwork_list
        RF=RF
        I_cut=I_cut
        print(f"RF is :{RF}")
        print(f"I_cut is :{I_cut}")
        def update_graph_parallel(G, V, D, dzdt, L, G_matrix, L_min, delta_t, static_edges, input_node_id,output_node_id, node_indices,RF=RF,I_cut=I_cut):
            graph_lock = threading.Lock()
            with ThreadPoolExecutor() as executor:
                    
                    futures = [
                        executor.submit(parallel_filament_update, idx, u, v,data,V,D,dzdt,input_node_id,output_node_id, L,G_matrix, L_min, delta_t,static_edges,G,RF=RF,I_cut=I_cut)
                        for idx, (u, v,data) in enumerate(edge_list)
                    ]
                    


                    for idx, (u, v, data) in enumerate(edge_list):
                        idx, G_id, I_new, dzdotbydz,Z_new = futures[idx].result()
                        with graph_lock:
                            # Z_old[idx, idx] = Z_new
                            G[u][v]['Y'] = G_id
                            G[u][v]['z'] = Z_new 
                            G_matrix[idx,idx]=G_id
                            # L[idx,idx]=Lij
                            I[idx]=I_new
                            dzdt[idx,idx]=dzdotbydz
            
            return G, G_matrix, I, dzdt
        for t in range(1,len(time1)):


            print(f"G_matrix is :{G_matrix}")

            boundary_conditions = {0: V_input, num_nodes-1: V_output}
            V=kaczmarz_method(M,G_matrix,V_input, V_output,V_initial,b)
            # V=kaczmarz_method(G, V_input,input_node,output_node)


            print(f"at time:{t}")
            print(f"voltage is :{V}")
            print(f"shape of Voltage vector is :{V.shape}")
            V_initial=V.copy()

            diag_MV = np.diag(np.matmul(M.T,V))

            lambda_reg=1e-6
            G, G_matrix, I, dzdt = update_graph_parallel(G, V, D, dzdt, L, G_matrix, L_min, delta_t, static_edges, input_node_id,output_node_id, node_indices,RF=RF,I_cut=I_cut)
            
            G_undirected = G.to_undirected()
            Rnetwork_list= nx.resistance_distance(G_undirected,input_node_id,output_node_id, weight='Y', invert_weight=False)   
            Ynetwork_list = 1/Rnetwork_list
            # Z_old=np.diag(np.diag(Z_old)/np.diag(D))
            print(f"Z_old is :{Z_old}")

            print(f"dzbydt is:{np.diag(dzdt)}")

            L=D*(Identity-Z_old)

            print(f"L is :{L}")
            print(f"maximum L is:{np.max(np.diag(L))}")
            print(f"minimum L is:{np.min(np.diag(L))}")
            print(f"maximum D is:{np.max(np.diag(D))}")
            print(f"minimum D is:{np.min(np.diag(D))}")
            print(f"max Z is :{np.max(np.diag(Z_old))}")
            print(f"min Z is :{np.min(np.diag(Z_old))}")
            print(f"max G is:{np.max(np.diag(G_matrix))}")
            print(f"min g is :{np.min(np.diag(G_matrix))}")

            dZ_dt_sparse = csr_matrix(dzdt)

            dZ_dt_storage_sparse2.append(dZ_dt_sparse)


            print(f"current is :{I}")

            total_current_output_node=np.sum(I)
            print(f"total output current is :{total_current_output_node}")

            power_dissipation = np.linalg.norm(np.matmul(G_matrix, np.matmul(M.T, V)**2))
            # cond=np.sum(np.diag(G_matrix))
            
            voltage.append(V[-1])
            # current.append(np.sum(np.diag(I)))
            # current.append(np.linalg.norm(I))
            current.append(total_current_output_node)
            power.append(power_dissipation)
            # conductance.append(cond)
            # conductance.append(np.linalg.norm(G_matrix))
            conductance.append(Ynetwork_list)
            timestamp.append(t)
            # time.append(t)
        endtime=time.time()
        total_time_taken=(endtime-starttime)
        print(f"time taken:{total_time_taken}")
        conductance_diff=np.diff(conductance)
        conductance_diff1=conductance_diff.tolist()
        conductance_diff1.append('')
        data = {
            "time": timestamp,
            "Current": current,
            "Power": power,
            "Conductance": conductance,
            "Conductance Diff": conductance_diff1
        }

        # Create a DataFrame from the dictionary
        df1 = pd.DataFrame(data)

        # Define the directory
        save_dir1 = "/home/jayanta/Download/code/material network/output_for_different_Icut_RF"

        # Ensure the directory exists
        os.makedirs(save_dir1, exist_ok=True)

        # Define the filename with variables
        filename1 = f"3d_horizontal_dzdotbydz_onlygrowthtime_Icut_{I_cut}_logic_eigenvalue_100sec_parallel_with_RF_{RF}_1V_beta_100_difference_using_cvxpy_boundary_condition_on_first_node_last_node_edge_exclude.csv"

        # Combine directory and filename
        save_path1 = os.path.join(save_dir1, filename1)
        df1.to_csv(save_path1, index=False)
        # Save the DataFrame to a CSV file
        # df.to_csv(f"dzdotbydz_onlygrowthtime_G_cut_Gon_IT0.000499999_Icut_{I_cut}_logic_eigenvalue_200sec_parallel_with_RF_{RF}_1V_beta_100_difference_using_cvxpy_boundary_condition on_first_node_last_node_edge_exclude.csv", index=False)


        for t in range(1,len(time1)):

            dz_storage_dense=dZ_dt_storage_sparse2[t].toarray()
            
            eigenvalues=[]
            for id in range(dz_storage_dense.shape[0]):
            


                eigenvalues.append(dz_storage_dense[id,id])

            eigenvalues_list.append(np.max(eigenvalues))
            mean_eigenvalue_list.append(np.mean(eigenvalues))
            min_eigenvalue_list.append(np.min(eigenvalues))

        data = {
            "time": timestamp,
            "max_eigenvalue":eigenvalues_list,
            "mean_eigenvalue":mean_eigenvalue_list,
            "min_eigevalue":min_eigenvalue_list


        }


        df2 = pd.DataFrame(data)

                
        # Define the directory
        save_dir2 = "/home/jayanta/Download/code/material network/output_for_different_Icut_RF"

        # Ensure the directory exists
        os.makedirs(save_dir2, exist_ok=True)

        filename2=f"3d_horizontal_dzdotbydz_onlygrowthtime_icut_{I_cut}_logic_RF_{RF}_dimless_max_eigenvalue_data_ineach_timestamp_100sec.csv"
        save_path2 = os.path.join(save_dir2, filename2)
        df2.to_csv(save_path2, index=False)

        print("Eigenvalues computed successfully.")
        # plt.figure(figsize=(10, 6))
        # for i, eigenvalues in enumerate(eigenvalues_list):
        #     plt.plot(np.real(eigenvalues), label=f'Time step {i}', marker='o')

        # plt.title('Eigenvalues Over Time')
        # plt.xlabel('Eigenvalue Index')
        # plt.ylabel('Eigenvalue Real Part')
        # plt.legend()
        # plt.grid(True)
        # plot_path = "/home/jayanta/myDownload/code/material network/dzdotbydz_onlygrowthtime_G_cut_Gon_IT0.000499999_icut_{I_cut}_logic_{RF}_dimless_Eigenvalues_Over_Time_for_60_percent_reduction_200sec_beta100.png"
        # plt.savefig(plot_path)
        # plt.close()
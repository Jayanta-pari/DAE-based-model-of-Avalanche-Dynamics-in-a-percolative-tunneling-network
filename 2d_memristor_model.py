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
start_time=time.time()
# import matplotlib.cm as cm
# cmap = cm.get_cmap('tab10')
# import matplotlib
# print(matplotlib.__version__)

# from cvxopt import matrix, solvers

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
rf=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
icut=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
for RF in rf:
    for I_cut in icut:

        # Global connection parameters (adjust as needed)
        d = 0.35      # minimum distance for connecting nodes
        max_d = 0.71  # maximum distance for connecting nodes
        start=time.time()

        def create_layered_graph(box_size, num_layers, points_per_layer):
        
            G = nx.DiGraph()
            layer_height = box_size / num_layers
            existing_nodes = set()
            # input_node
            # output_node
            
                # Add the extra input node above the top layer.
            input_y = box_size + layer_height  # Placed above the top layer
            input_node = (box_size / 2, input_y)

            output_y = -layer_height  # Placed below the bottom layer
            output_node = (box_size / 2, output_y)
            G.add_node(input_node, layer=-1)
            # Create nodes for each layer (each layer gets `points_per_layer` nodes)
            for layer in range(num_layers):
                y_offset = box_size - (layer + 1) * layer_height
                for _ in range(points_per_layer):
                    while True:
                        x = random.uniform(0, box_size)
                        y = random.uniform(y_offset, y_offset + layer_height)
                        new_node = (x, y)
                        # Ensure the new node is at least distance d away from all other nodes
                        if all(np.linalg.norm(np.array(new_node) - np.array(n)) > d for n in existing_nodes):
                            G.add_node(new_node, layer=layer)
                            existing_nodes.add(new_node)
                            break


            G.add_node(output_node, layer=num_layers)


            first_layer_nodes = [n for n, data in G.nodes(data=True) if data.get('layer') == 0]
            for node in first_layer_nodes:
                distance = np.linalg.norm(np.array(input_node) - np.array(node))
                # if d <= distance <= max_d:
                G.add_edge(input_node, node, distance=distance)

            # Connect all nodes in the last layer to the output node
            last_layer_nodes = [n for n, data in G.nodes(data=True) if data.get('layer') == num_layers - 1]
            for node in last_layer_nodes:
                distance = np.linalg.norm(np.array(node) - np.array(output_node))
                # if d <= distance <= max_d:
                G.add_edge(node, output_node, distance=distance)

            # Connect nodes within each layer (directed edges)
            for layer in range(num_layers):
                layer_nodes = [n for n, data in G.nodes(data=True) if data.get('layer') == layer]
                for i in range(len(layer_nodes)):
                    for j in range(i + 1, len(layer_nodes)):
                        node1, node2 = layer_nodes[i], layer_nodes[j]
                        distance = np.linalg.norm(np.array(node1) - np.array(node2))
                        if d <=distance <= max_d:
                            # Decide direction based on x-coordinate (or y if tie)
                            # if (node1[0] < node2[0]) or (node1[0] == node2[0] and node1[1] > node2[1]):
                            if (node1[1]>node2[1]):
                                G.add_edge(node1, node2, distance=distance)
                            else:
                                G.add_edge(node2, node1, distance=distance)

            # Connect nodes between adjacent layers
            for layer in range(num_layers - 1):
                current_layer_nodes = [n for n, data in G.nodes(data=True) if data.get('layer') == layer]
                next_layer_nodes = [n for n, data in G.nodes(data=True) if data.get('layer') == layer + 1]
                for node1 in current_layer_nodes:
                    for node2 in next_layer_nodes:
                        distance = np.linalg.norm(np.array(node1) - np.array(node2))
                        if d <=distance <= max_d:
                            G.add_edge(node1, node2, distance=distance)

            # Build a position dictionary for plotting (each node's position is its coordinate)
            pos = {node: node for node in G.nodes()}

            # Relabel nodes to integers while preserving their original position in the node attribute 'pos'.
            G = nx.convert_node_labels_to_integers(G, label_attribute='pos')

            return G, pos

        def plot_graph(G, pos):
            """
            Plots the directed graph `G` using the positions in `pos`. 
            Nodes are colored by their layer:
            - 'input' node: red
            - 'output' node: blue
            - Intermediate layers: a colormap based on the layer index.
            """
            # Build a mapping from node to its layer (recall that after relabeling, the original coordinate is stored in 'pos')
            layers = {}
            for node, data in G.nodes(data=True):
                # The original layer information is lost in pos, so we extract it from the stored 'pos' attribute if needed.
                # Since we didn't store the layer in the 'pos' attribute, we retrieve it from our original G (if available)
                # Instead, we could save the layer in another attribute if necessary.
                # For demonstration, we'll assume the layer info is stored in data under key 'layer'
                layers[node] = data.get('layer')

            # Choose colors for nodes based on their layer.
            node_colors = []
            for node in G.nodes():
                layer = layers[node]
                if layer == 'input':
                    node_colors.append('red')
                elif layer == 'output':
                    node_colors.append('blue')
                else:
                    # Use a colormap for numeric layers.
                    # Normalize the layer index to [0, 1] (assuming layers from 0 to max_layer)
                    norm_val = layer / (max(l for l in layers.values() if isinstance(l, int)) or 1)
                    # Convert normalized value to a color using plt.cm.viridis
                    node_colors.append(plt.cm.viridis(norm_val))

            plt.figure(figsize=(10, 8))
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.9)
            # Draw edges with arrows
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
            
            # Draw labels (optional)
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

            plt.title("Layered Directed Graph")
            plt.axis("equal")
            plt.axis("off")
            plt.show()

        # ------------------------------
        # Example usage:
        box_size = 10      # Size of the box in which nodes are placed
        num_layers = 14     # Number of layers (not including input/output nodes)
        points_per_layer = 30  # Number of nodes in each layer

        # Create the graph and get node positions
        G, pos = create_layered_graph(box_size, num_layers, points_per_layer)


        pos_relabel = {node: data['pos'] for node, data in G.nodes(data=True)}

        # Plot the graph
        plot_graph(G, pos_relabel)


        input_node = None
        output_node=None
        for node, data in G.nodes(data=True):
            if data.get('layer') == -1:  # input node has layer -1
                input_node = node
            if data.get('layer') == num_layers:  # output node has layer num_layers
                output_node = node
                break
        print(f"output data is:{output_node}")
        print(f"input data is:{input_node}")

        output_degree = G.degree[input_node]
        print(f"The number of edges connected to the output node: {output_degree}")
        edge_list = [(u, v,data) for u, v, data in G.edges(data=True)]
        # print(edge_list)
        num_nodes = G.number_of_nodes()
        num_edges = len(edge_list)
        
        def compute_matrices(G,edge_list):
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
                if u not in (input_node,output_node) and v not in (input_node,output_node):
                    G[u][v]['flag']=False
                    G[u][v]['Y']=G0*np.exp(-beta*D[edge_idx, edge_idx])
                    G_matrix[edge_idx, edge_idx]= G[u][v]['Y']
                    G[u][v]['z']=0
                else:
                    G[u][v]['flag']=True
                    G[u][v]['Y']=1
                    G_matrix[edge_idx, edge_idx]= G[u][v]['Y']
                    G[u][v]['z']=D[edge_idx, edge_idx]
            
            return M, D,G_matrix,G

        # Generate graph and compute matrices
        # box_size = 10
        # num_layers = 9
        # points_per_layer = 50
        # G = create_layered_graph(box_size, num_layers, points_per_layer)
        M, D,G_matrix,G = compute_matrices(G,edge_list)

        # Visualization
        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        nx.draw(G, pos, node_size=5, arrowsize=8, with_labels=False)
        plt.title('Directed Graph with Integer Nodes')

        plt.subplot(122)
        plt.spy(D)
        plt.title('Distance Matrix (D)')
        plt.show()

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
        print(f"output data is:{output_node}")
        output_degree = G.degree[input_node]
        print(f"The number of edges connected to the output node: {output_degree}")

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
            if node_indices[u] in (input_node,output_node) or node_indices[v] in (input_node,output_node):

                static_edges.add((u, v)) 
        #         G[u][v]['Y']=1
        #         G_matrix[idx, idx]= G[u][v]['Y']
                #G[u][v]['flag']=True
                #static_edges.add((v, u)) 
        t = 0.0
        T = 200


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
            A=A/np.linalg.norm(A,'fro')
            m, n = A.shape
            print(f"rank of matrix is:{np.linalg.matrix_rank(A)}")

        
            V_max=V_input
            V_min=V_output
            X = cp.Variable(n)
            

            constraints = [
            
            X[0] == V_input,# Boundary condition at X[0]
        
            X[-1] == V_output,  # Boundary condition at X[-1]

            # X<=V_max,
            X>=V_min
            ]
            #objective=cp.Minimize(cp.norm(A @ X,2))
            objective=cp.Minimize(cp.sum_squares(A @ X))
            problem = cp.Problem(objective, constraints)

            
            problem.solve(verbose=True,solver=cp.ECOS,reltol=1e-9)



            return X.value


            
            
        def update_filament_growth(idx,u,v,G, D_idx,dzdotbydz,V_jn,Lij,G_id, V_set=0.01, V_reset=0.005, L_max=np.max(D), beta=50, mu=0.346,kappa=0.038, delta_t=0.01,RF=RF,I_cut=I_cut):
            G_cut=0.8
            G_joule=0.01
            epsilon = 1e-6 
            # if D_idx>0:
            max_d=0.69
            I_cut=I_cut*(0.0001)
            G_cut=0.001
            Gon=0.933
            RF=RF
            # IT=0.000499999
            cur_factor=0.00001
            IT=round(I_cut*cur_factor)



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
                    if I_new>=I_cut:
                    #if G_id>=(0.01*Gon):
                        G[u][v]['flag']=True
                    #     # Z_old=D_idx-Lij
                    #     Z_old=0.999
                else:
                    # G[u][v]['Y']=G0*np.exp(-beta*(D_idx-(G[u][v]['z'])))
                    G[u][v]['Y']=G0*np.exp(-beta*D_idx*(1-(G[u][v]['z'])))
                    G_id=G[u][v]['Y']+0.00001
                    I_new=V_jn*G_id
                    # if G_id>=(G_cut):
                    #    G[u][v]['flag']=True
                    #dzdotbydz=((((mu*V_jn)/((D_idx**2)*(1-G[u][v]['z'])**2))) -kappa)
                    dzdotbydz=1-((4*mu*V_jn)/(kappa*D_idx**2))
        
                # return G_id,I_new,dzdotbydz
            else:
                # print("hi i am decreasing")
                # Z_old= max(0,Z_old-(1/RF))
                G[u][v]['z']=G[u][v]['z'] - ((1/RF)*G[u][v]['z'])
                # Z_old=Z_old/D_idx
                # G_id=G0*np.exp(-beta*D_idx*(1-Z_old))
                G[u][v]['Y']=G0*np.exp(-beta*D_idx*(1-(G[u][v]['z'])))
                # G[u][v]['Y']=G0*np.exp(-beta*(D_idx-(G[u][v]['z'])))
                G_id=G[u][v]['Y']
                I_new=V_jn*G_id
                dzdotbydz=1-((4*mu*V_jn)/(kappa*D_idx**2))
                if I_new<= abs(I_cut-IT):

                    G[u][v]['flag']=False



            return G_id,I_new,dzdotbydz,G[u][v]['z']




        def parallel_filament_update(idx, u, v,data,V,D,dzdt, L,G_matrix, L_min, delta_t,static_edges,G,RF=RF,I_cut=I_cut):
            

            # if u in (input_node,output_node) or v in (input_node,output_node):
            # if u == input_node or v == input_node or u == output_node or v == output_node:
            # continue  # Skip input/output edges

                    
                # return idx, Z_old[idx, idx], 1,0,0
        
            

            # if node_indices[u] not in (input_node,output_node) and node_indices[v] not in (input_node,output_node):
            if u not in (input_node,output_node) and v not in (input_node,output_node):
                V_jn = abs(V[node_indices[u]] - V[node_indices[v]])

                G_id,I_new,dzdotbydz,Z_new = update_filament_growth(idx,u,v,G,D[idx,idx],dzdt[idx,idx],V_jn,L[idx,idx],G_matrix[idx,idx], L_max=np.max(D),RF=RF,I_cut=I_cut)

                # I_new=G_matrix[idx,idx]*(V_jn)

            

                return idx,G_id,I_new,dzdotbydz,Z_new

            else:
                return idx, 1,0,0,G[u][v]['z']
            



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
        Rnetwork_list= nx.resistance_distance(G_undirected,input_node, output_node, weight='Y', invert_weight=False)   
        Ynetwork_list = 1/Rnetwork_list
        RF=RF
        I_cut=I_cut
        print(f"RF is :{RF}")
        print(f"I_cut is :{I_cut}")
        def update_graph_parallel(G, V, D, dzdt, L, G_matrix, L_min, delta_t, static_edges, input_node, output_node, node_indices,RF=RF,I_cut=I_cut):
            graph_lock = threading.Lock()
            with ThreadPoolExecutor() as executor:
                    
                    futures = [
                        executor.submit(parallel_filament_update, idx, u, v,data,V,D,dzdt, L,G_matrix, L_min, delta_t,static_edges,G,RF=RF,I_cut=I_cut)
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
            G, G_matrix, I, dzdt = update_graph_parallel(G, V, D, dzdt, L, G_matrix, L_min, delta_t, static_edges, input_node, output_node, node_indices,RF=RF,I_cut=I_cut)
            
            G_undirected = G.to_undirected()
            Rnetwork_list= nx.resistance_distance(G_undirected,input_node, output_node, weight='Y', invert_weight=False)   
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
        end=time.time()
        total_time_taken=(end-start)
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
        filename1 = f"stability_dzdotbydz_onlygrowthtime_G_cut_Gon_IT0.000499999_Icut_{I_cut}_logic_eigenvalue_200sec_parallel_with_RF_{RF}_1V_beta_100_difference_using_cvxpy_boundary_condition_on_first_node_last_node_edge_exclude.csv"

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

        filename2=f"stability_dzdotbydz_onlygrowthtime_G_cut_Gon_IT0.000499999_icut_{I_cut}_logic_RF_{RF}_dimless_max_eigenvalue_data_ineach_timestamp_200sec.csv"
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
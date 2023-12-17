import basix
import numpy as onp
import jax.numpy as np
import jax
from functools import partial
from scipy.sparse import csc_matrix
import scipy.sparse
import time
from itertools import combinations
import os,sys

from jax.config import config
config.update("jax_enable_x64", True)

def GaussSet(Gauss_num = 2, cuda=False):
    if Gauss_num == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
       
    elif Gauss_num == 3:
        Gauss_Weight1D = [0.55555556, 0.88888889, 0.55555556]
        Gauss_Point1D = [-0.7745966, 0, 0.7745966]
       
        
    elif Gauss_num == 4:
        Gauss_Weight1D = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        Gauss_Point1D = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]

    elif Gauss_num == 6: # double checked, 16 digits
        Gauss_Weight1D = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        Gauss_Point1D = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 
                         0.2386191860831969, 0.6612093864662645, 0.9324695142031521]

       
    elif Gauss_num == 8: # double checked, 20 digits
        Gauss_Weight1D=[0.10122853629037625915, 0.22238103445337447054, 0.31370664587788728733, 0.36268378337836198296,
                        0.36268378337836198296, 0.31370664587788728733, 0.22238103445337447054,0.10122853629037625915]
        Gauss_Point1D=[-0.960289856497536231684, -0.796666477413626739592,-0.525532409916328985818, -0.183434642495649804939,
                        0.183434642495649804939,  0.525532409916328985818, 0.796666477413626739592,  0.960289856497536231684]
        
    elif Gauss_num == 10:
        Gauss_Weight1D=[0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        Gauss_Point1D=[-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,  
                        0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717]
        
    elif Gauss_num == 20:
        Gauss_Weight1D=[0.017614007, 0.04060143, 0.062672048, 0.083276742,0.10193012, 0.118194532,0.131688638,
                        0.142096109, 0.149172986, 0.152753387,0.152753387,0.149172986, 0.142096109, 0.131688638,
                        0.118194532,0.10193012, 0.083276742,0.062672048,0.04060143,0.017614007]
            
        Gauss_Point1D=[-0.993128599, -0.963971927, -0.912234428, -0.839116972, -0.746331906, -0.636053681,
                        -0.510867002, -0.373706089, -0.227785851, -0.076526521, 0.076526521, 0.227785851,
                        0.373706089, 0.510867002, 0.636053681, 0.746331906, 0.839116972, 0.912234428, 0.963971927, 0.993128599]
    
    return Gauss_Weight1D, Gauss_Point1D

def uniform_mesh(d1, nx, element_type, regular_mesh_bool):
    
    if element_type == 'D1LN2N': # 1D 2-node linear element
    
        dim = 1 # problem dimension
        nnode = nx+1 # number of nodes
        nelem = nx # number of elements
        nodes_per_elem = 2 # nodes per elements

        iffix = onp.zeros(nnode, dtype=onp.int32)

        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction

        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nx+2):
            XY[n,0] = (i-1)*a # for x values
            # if i == 1 or i == nx+1: # boundary
            if i == 1 or i == nx+1: # boundary
                iffix[n] = 1
            n += 1
            
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, nx+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = j 
    
    elif element_type == 'D1LN3N': # 1D 3-node quadratic element
    
        dim = 1 # problem dimension
        nnode = 2*nx+1 # number of nodes
        nelem = nx # number of elements
        nodes_per_elem = 3 # nodes per elements

        iffix = onp.zeros(nnode, dtype=onp.int32)

        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction
        a2 = a/2

        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nx+2):
            XY[n,0] = (i-1)*a # for x values
            if i == 1 or i == nx+1: # boundary
                iffix[n] = 1
            n += 1

        # x-axis center nodes
        for i in range(1, nx+1): # because of x-axis center nodes
            XY[n,0] = (i-1)*a + a2 # for x values
            n += 1

        ## elements ##
        x_cen_start = nx+1
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, nx+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = x_cen_start + j-1
            Elem_nodes[j-1, 2] = j
        
        
    elif element_type == 'D1LN4N': # 1D 3-node cubic element
        
        dim = 1 # problem dimension
        nnode = 3*nx+1 # number of nodes
        nelem = nx # number of elements
        nodes_per_elem = 4 # nodes per elements

        iffix = onp.zeros(nnode, dtype=onp.int32)

        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction
        a3 = a/3

        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nx+2):
            XY[n,0] = (i-1)*a # for x values
            if i == 1 or i == nx+1: # boundary
                iffix[n] = 1
            n += 1

        # x-axis center nodes
        for i in range(1, nx+1): # because of x-axis center nodes
            XY[n,0] = (i-1)*a + a3 # for x values
            XY[n+1,0] = (i-1)*a + 2*a3 # for x values
            n += 2


        ## elements ##
        x_cen_start = nx+1
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, nx+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = x_cen_start + 2*(j-1)
            Elem_nodes[j-1, 2] = x_cen_start + 2*(j-1)+1
            Elem_nodes[j-1, 3] = j
            
            
    elif element_type == 'D2QU4N': # 2D 4-node linear element
    
        dim = 2 # problem dimension
        d2, ny = d1, nx
        nnode = (nx+1)*(ny+1) # number of nodes
        nelem = nx*ny # number of elements
        nodes_per_elem = 4 # nodes per elements

        iffix = onp.zeros(nnode, dtype=onp.int32)

        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction
        b = d2/ny # increment in the y directin

        n = 0 # This will allow us to go through rows in NL
        for j in range(1, ny+2):
            for i in range(1, nx+2):
                if regular_mesh_bool:
                    XY[n,0] = (i-1)*a # for x values
                    XY[n,1] = (j-1)*b # for y values
                else:
                    XY[n,0] = (i-1)*a + onp.random.normal(0,0.1,1)*a
                    XY[n,1] = (j-1)*b + onp.random.normal(0,0.1,1)*b
                if i == 1 or i == nx+1 or j == 1 or j == ny+1: # boundary
                    iffix[n] = 1
                n += 1
                
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                elem_idx = nx*(j-1)+i-1
                if i == 1:
                    Elem_nodes[elem_idx, 0] = (nx+1)*(j-1)
                    Elem_nodes[elem_idx, 1] = (nx+1)*(j-1) + 1
                    Elem_nodes[elem_idx, 2] = (nx+1)*(j) + 1
                    Elem_nodes[elem_idx, 3] = (nx+1)*(j)
                else:
                    Elem_nodes[elem_idx, 0] = Elem_nodes[elem_idx-1, 0] + 1
                    Elem_nodes[elem_idx, 1] = Elem_nodes[elem_idx-1, 1] + 1
                    Elem_nodes[elem_idx, 2] = Elem_nodes[elem_idx-1, 2] + 1
                    Elem_nodes[elem_idx, 3] = Elem_nodes[elem_idx-1, 3] + 1
            
    elif element_type == 'D3BR8N': # 3D 8-node elemenet
        dim = 3 # problem dimension
        d2, ny = d1, nx
        d3, nz = d1, nx
        nnode = (nx+1)*(ny+1)*(nz+1) # number of nodes
        nelem = nx*ny*nz # number of elements
        nodes_per_elem = 8 # nodes per elements

        iffix = onp.zeros(nnode, dtype=onp.int32)

        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction
        b = d2/ny # increment in the y directin
        c = d3/nz # increment in the z directin

        n = 0 # This will allow us to go through rows in NL
        for k in range(1, nz+2):
            for j in range(1, ny+2):
                for i in range(1, nx+2):
                    XY[n,0] = (i-1)*a # for x values
                    XY[n,1] = (j-1)*b # for y values
                    XY[n,2] = (k-1)*c # for z values
                    if i == 1 or i == nx+1 or j == 1 or j == ny+1 or k == 1 or k == nz+1: # boundary
                        iffix[n] = 1
                    n += 1
                
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for k in range(1, nz+1):
            for j in range(1, ny+1):
                for i in range(1, nx+1):
                    elem_idx = nx*ny*(k-1)+nx*(j-1)+i-1
                    if i == 1:
                        Elem_nodes[elem_idx, 0] = (nx+1)*(ny+1)*(k) + (nx+1)*(j)
                        Elem_nodes[elem_idx, 1] = (nx+1)*(ny+1)*(k) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx, 2] = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx, 3] = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j)
                        Elem_nodes[elem_idx, 4] = Elem_nodes[elem_idx, 0]+1
                        Elem_nodes[elem_idx, 5] = Elem_nodes[elem_idx, 1]+1
                        Elem_nodes[elem_idx, 6] = Elem_nodes[elem_idx, 2]+1
                        Elem_nodes[elem_idx, 7] = Elem_nodes[elem_idx, 3]+1
                    else:
                        Elem_nodes[elem_idx, 0] = Elem_nodes[elem_idx-1, 4]
                        Elem_nodes[elem_idx, 1] = Elem_nodes[elem_idx-1, 5]
                        Elem_nodes[elem_idx, 2] = Elem_nodes[elem_idx-1, 6]
                        Elem_nodes[elem_idx, 3] = Elem_nodes[elem_idx-1, 7]
                        Elem_nodes[elem_idx, 4] = Elem_nodes[elem_idx, 0]+1
                        Elem_nodes[elem_idx, 5] = Elem_nodes[elem_idx, 1]+1
                        Elem_nodes[elem_idx, 6] = Elem_nodes[elem_idx, 2]+1
                        Elem_nodes[elem_idx, 7] = Elem_nodes[elem_idx, 3]+1
    
    elif element_type == 'D3BR20N': # 3D 20-node quadratic serendipity elemenet
        dim = 3 # problem dimension
        d2, ny = d1, nx
        d3, nz = d1, nx
        nnode = (nx+1)*(ny+1)*(nz+1) + (nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz) # number of nodes
        nelem = nx*ny*nz # number of elements
        nodes_per_elem = 20 # nodes per elements
    
        iffix = onp.zeros(nnode, dtype=onp.int32)
    
        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        a = d1/nx # increment in the x direction
        b = d2/ny # increment in the y directin
        c = d3/nz # increment in the z directin
        a2 = a/2 # center nodes increment
        b2 = b/2
        c2 = c/2
    
        # corner nodes
        n = 0 # This will allow us to go through rows in NL
        for k in range(1, nz+2):
            for j in range(1, ny+2):
                for i in range(1, nx+2):
                    XY[n,0] = (i-1)*a # for x values
                    XY[n,1] = (j-1)*b # for y values
                    XY[n,2] = (k-1)*c # for z values
                    if i == 1 or i == nx+1 or j == 1 or j == ny+1 or k == 1 or k == nz+1: # boundary
                        iffix[n] = 1
                    n += 1
                    
        # x-axis center nodes
        for k in range(1, nz+2):
            for j in range(1, ny+2):
                for i in range(1, nx+1): # because of x-axis center nodes
                    XY[n,0] = (i-1)*a + a2 # for x values
                    XY[n,1] = (j-1)*b # for y values
                    XY[n,2] = (k-1)*c # for z values
                    if j == 1 or j == ny+1 or k == 1 or k == nz+1: # boundary
                        iffix[n] = 1
                    n += 1
                    
        # y-axis center nodes
        for k in range(1, nz+2):
            for j in range(1, ny+1): # because of x-axis center nodes
                for i in range(1, nx+2):
                    XY[n,0] = (i-1)*a # for x values
                    XY[n,1] = (j-1)*b + b2 # for y values
                    XY[n,2] = (k-1)*c # for z values
                    if i == 1 or i == ny+1 or k == 1 or k == nz+1: # boundary
                        iffix[n] = 1
                    n += 1
                    
        # z-axis center nodes
        for k in range(1, nz+1): # because of x-axis center nodes
            for j in range(1, ny+2):
                for i in range(1, nx+2):
                    XY[n,0] = (i-1)*a # for x values
                    XY[n,1] = (j-1)*b # for y values
                    XY[n,2] = (k-1)*c + c2 # for z values
                    if i == 1 or i == ny+1 or j == 1 or j == ny+1: # boundary
                        iffix[n] = 1
                    n += 1
    
                
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        x_cen_start = (nx+1)*(ny+1)*(nz+1)
        y_cen_start = (nx+1)*(ny+1)*(nz+1) + (nx)*(ny+1)*(nz+1)
        z_cen_start = (nx+1)*(ny+1)*(nz+1) + (nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1)
    
        for k in range(1, nz+1):
            for j in range(1, ny+1):
                for i in range(1, nx+1):
                    elem_idx = nx*ny*(k-1)+nx*(j-1)+i-1
                    if i == 1:
                        Elem_nodes[elem_idx, 0] = (nx+1)*(ny+1)*(k  ) + (nx+1)*(j)
                        Elem_nodes[elem_idx, 1] = (nx+1)*(ny+1)*(k  ) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx, 2] = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx, 3] = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j)
                        Elem_nodes[elem_idx, 4] = Elem_nodes[elem_idx, 0]+1
                        Elem_nodes[elem_idx, 5] = Elem_nodes[elem_idx, 1]+1
                        Elem_nodes[elem_idx, 6] = Elem_nodes[elem_idx, 2]+1
                        Elem_nodes[elem_idx, 7] = Elem_nodes[elem_idx, 3]+1
                        
                        Elem_nodes[elem_idx, 8] = x_cen_start + (nx)*(ny+1)*(k  ) + (nx)*(j)
                        Elem_nodes[elem_idx, 9] = x_cen_start + (nx)*(ny+1)*(k  ) + (nx)*(j-1)
                        Elem_nodes[elem_idx,10] = x_cen_start + (nx)*(ny+1)*(k-1) + (nx)*(j-1)
                        Elem_nodes[elem_idx,11] = x_cen_start + (nx)*(ny+1)*(k-1) + (nx)*(j)
                        
                        Elem_nodes[elem_idx,12] = y_cen_start + (nx)*(ny+1)*(k  ) + (nx+1)*(j-1) + 1
                        Elem_nodes[elem_idx,13] = y_cen_start + (nx)*(ny+1)*(k  ) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx,14] = y_cen_start + (nx)*(ny+1)*(k-1) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx,15] = y_cen_start + (nx)*(ny+1)*(k-1) + (nx+1)*(j-1) + 1
                        
                        Elem_nodes[elem_idx,16] = z_cen_start + (nx+1)*(ny+1)*(k-1) + (nx+1)*(j  ) + 1
                        Elem_nodes[elem_idx,17] = z_cen_start + (nx+1)*(ny+1)*(k-1) + (nx+1)*(j  )
                        Elem_nodes[elem_idx,18] = z_cen_start + (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1)
                        Elem_nodes[elem_idx,19] = z_cen_start + (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1) + 1
                        
                        
                    else:
                        Elem_nodes[elem_idx, 0] = Elem_nodes[elem_idx-1, 4]
                        Elem_nodes[elem_idx, 1] = Elem_nodes[elem_idx-1, 5]
                        Elem_nodes[elem_idx, 2] = Elem_nodes[elem_idx-1, 6]
                        Elem_nodes[elem_idx, 3] = Elem_nodes[elem_idx-1, 7]
                        Elem_nodes[elem_idx, 4] = Elem_nodes[elem_idx, 0]+1
                        Elem_nodes[elem_idx, 5] = Elem_nodes[elem_idx, 1]+1
                        Elem_nodes[elem_idx, 6] = Elem_nodes[elem_idx, 2]+1
                        Elem_nodes[elem_idx, 7] = Elem_nodes[elem_idx, 3]+1
                        
                        Elem_nodes[elem_idx, 8] = Elem_nodes[elem_idx-1, 8]+1
                        Elem_nodes[elem_idx, 9] = Elem_nodes[elem_idx-1, 9]+1
                        Elem_nodes[elem_idx,10] = Elem_nodes[elem_idx-1,10]+1
                        Elem_nodes[elem_idx,11] = Elem_nodes[elem_idx-1,11]+1
                        
                        Elem_nodes[elem_idx,12] = Elem_nodes[elem_idx-1,12]+1
                        Elem_nodes[elem_idx,13] = Elem_nodes[elem_idx-1,13]+1
                        Elem_nodes[elem_idx,14] = Elem_nodes[elem_idx-1,14]+1
                        Elem_nodes[elem_idx,15] = Elem_nodes[elem_idx-1,15]+1
                        
                        Elem_nodes[elem_idx,16] = Elem_nodes[elem_idx-1,16]+1
                        Elem_nodes[elem_idx,17] = Elem_nodes[elem_idx-1,17]+1
                        Elem_nodes[elem_idx,18] = Elem_nodes[elem_idx-1,18]+1
                        Elem_nodes[elem_idx,19] = Elem_nodes[elem_idx-1,19]+1
        # EL += 1 # indexing has already been done in python index
    
    # EL -= 1 # python indexing
    dof_global = len(iffix)
        
    return XY, Elem_nodes, iffix, nnode, nelem, dof_global

def get_quad_points(Gauss_num, dim):
    # This function is compatible with FEM and CFEM
    # surf_bool: indicates whether an element is regular element or surface element
    # surf_dir: surface direction. it can be 0, 1, 2 for x, y, z, respectively
    # surf_val: can be +1 or -1. indicates the surface coord. in natural coord.
    
    Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_num)
    quad_points, quad_weights = [], []
    
    for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
        if dim == 1:
            quad_points.append([ipoint])
            quad_weights.append(iweight)
        else:
            for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                if dim == 2:
                    quad_points.append([ipoint, jpoint])
                    quad_weights.append(iweight * jweight)
                else: # dim == 3
                    for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                        quad_points.append([ipoint, jpoint, kpoint])
                        quad_weights.append(iweight * jweight * kweight)
    
    quad_points = np.array(quad_points) # (quad_degree*dim, dim)
    quad_weights = np.array(quad_weights) # (quad_degree,)
    return quad_points, quad_weights


def get_shape_val_functions(elem_type):
    """Hard-coded first order shape functions in the parent domain.
    Important: f1-f8 order must match "self.cells" by gmsh file!
    """
    ############ 1D ##################
    if elem_type == 'D1LN2N': # 1D linear element
        f1 = lambda x: 1./2.*(1 - x[0])
        f2 = lambda x: 1./2.*(1 + x[0]) 
        shape_fun = [f1, f2]
    
    elif elem_type == 'D1LN3N': # 1D quadratic element
        # ref: https://kratos-wiki.cimne.upc.edu/index.php/One-dimensional_Shape_Functions    
        f1 = lambda x: 1./2.*x[0]*(x[0]-1)
        f2 = lambda x: (1+x[0])*(1-x[0])
        f3 = lambda x: 1./2.*x[0]*(x[0]+1) 
        shape_fun = [f1, f2, f3]
    
    elif elem_type == 'D1LN4N': # 1D cubic element
        # ref: https://kratos-wiki.cimne.upc.edu/index.php/One-dimensional_Shape_Functions    
        f1 = lambda x: -9./16.*(x[0]+1./3.)*(x[0]-1./3.)*(x[0]   -1.)
        f2 = lambda x: 27./16.*(x[0]   +1.)*(x[0]-1./3.)*(x[0]   -1.)
        f3 = lambda x:-27./16.*(x[0]   +1.)*(x[0]+1./3.)*(x[0]   -1.)
        f4 = lambda x:  9./16.*(x[0]   +1.)*(x[0]+1./3.)*(x[0]-1./3.)
        shape_fun = [f1, f2, f3, f4]
        
        
    ############# 2D #########################
    elif elem_type == 'QUAD4': # 2D linear element
        f1 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1)) # (-1, -1)
        f2 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1)) # (-1,  1)
        shape_fun = [f1, f2, f3, f4]
        
    
    elif elem_type == 'D2QU8N': # 2D quadratic serendipity element
        f1 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 - x[0]*(-1) - x[1]*(-1)) # (-1, -1)
        f2 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 - x[0]*( 1) - x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 - x[0]*( 1) - x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 - x[0]*(-1) - x[1]*( 1)) # (-1,  1)
        f5 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*(-1)) # ( 0, -1)
        f6 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*( 1)) # ( 1,  0)
        f7 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*( 1)) # ( 0,  1)
        f8 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*(-1)) # (-1,  0)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]
        
    elif elem_type == 'D2QU9N': # 2D quadratic element
        f1 = lambda x:  1./4. * x[0] * (1 - x[0]) * x[1] * (1 - x[1])
        f2 = lambda x: -1./4. * x[0] * (1 + x[0]) * x[1] * (1 - x[1])
        f3 = lambda x:  1./4. * x[0] * (1 + x[0]) * x[1] * (1 + x[1])
        f4 = lambda x: -1./4. * x[0] * (1 - x[0]) * x[1] * (1 + x[1])
        f5 = lambda x: -1./2. * (1 + x[0]) * (1 - x[0]) * x[1] * (1 - x[1])
        f6 = lambda x:  1./2. * x[0] * (1 + x[0]) * (1 + x[1]) * (1 - x[1])
        f7 = lambda x:  1./2. * (1 + x[0]) * (1 - x[0]) * x[1] * (1 + x[1])
        f8 = lambda x: -1./2. * x[0] * (1 - x[0]) * (1 + x[1]) * (1 - x[1])
        f9 = lambda x: (1 - x[0]**2) * (1 - x[1]**2)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        
    elif elem_type == 'D2QU12N': # 2D cubic serendipity element
        f1 = lambda x: -1./32.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(10 - 9*x[0]**2 - 9*x[1]**2) # (-1, -1)
        f2 = lambda x: -1./32.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(10 - 9*x[0]**2 - 9*x[1]**2) # ( 1, -1)
        f3 = lambda x: -1./32.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(10 - 9*x[0]**2 - 9*x[1]**2) # ( 1,  1)
        f4 = lambda x: -1./32.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(10 - 9*x[0]**2 - 9*x[1]**2) # (-1,  1)
        f5 = lambda x:  9./32.*(1 - x[0]**2)*(1 + x[0]/(-1/3))*(1 + x[1]*(-1)) # (-1/3,  -1)
        f7 = lambda x:  9./32.*(1 - x[0]**2)*(1 + x[0]/( 1/3))*(1 + x[1]*( 1)) # ( 1/3,   1)
        f9 = lambda x:  9./32.*(1 - x[0]**2)*(1 + x[0]/( 1/3))*(1 + x[1]*(-1)) # ( 1/3,  -1)
        f11= lambda x:  9./32.*(1 - x[0]**2)*(1 + x[0]/(-1/3))*(1 + x[1]*( 1)) # (-1/3,   1)
        f6 = lambda x:  9./32.*(1 - x[1]**2)*(1 + x[1]/(-1/3))*(1 + x[0]*( 1)) # ( 1, -1/3)
        f8 = lambda x:  9./32.*(1 - x[1]**2)*(1 + x[1]/( 1/3))*(1 + x[0]*(-1)) # (-1,  1/3)
        f10= lambda x:  9./32.*(1 - x[1]**2)*(1 + x[1]/( 1/3))*(1 + x[0]*( 1)) # ( 1,  1/3)
        f12= lambda x:  9./32.*(1 - x[1]**2)*(1 + x[1]/(-1/3))*(1 + x[0]*(-1)) # (-1, -1/3)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
        
    elif elem_type == 'D2QU16N': # 2D cubic element
        f1 = lambda x: 3**4/2**8 * (1/3 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1/3 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f2 = lambda x: 3**4/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1/3 - x[0]) * (1/3 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f3 = lambda x: 3**4/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1/3 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1/3 - x[1])
        f4 = lambda x: 3**4/2**8 * (1/3 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1/3 - x[1])
        f5 = lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1/3 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f6 = lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1/3 - x[0]) * (1 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f7 = lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1/3 - x[1])
        f8 = lambda x: -3**5/2**8 * (1/3 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1 - x[1])
        f9 = lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1 - x[0]) * (1/3 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f10= lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1/3 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1 - x[1])
        f11= lambda x: -3**5/2**8 * (1 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1/3 - x[1])
        f12= lambda x: -3**5/2**8 * (1/3 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f13= lambda x: 3**6/2**8 * (1 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f14= lambda x: 3**6/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 - x[1]) * (1 - x[1])
        f15= lambda x: 3**6/2**8 * (1 + x[0]) * (1/3 + x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1 - x[1])
        f16= lambda x: 3**6/2**8 * (1 + x[0]) * (1/3 - x[0]) * (1 - x[0]) * (1 + x[1]) * (1/3 + x[1]) * (1 - x[1])
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]
    
    elif elem_type == 'CPE4':
        f1 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1)) # (-1, -1)
        f2 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1)) # (-1,  1)
        shape_fun = [f1, f2, f3, f4]
        
    elif elem_type == 'CPE8':
        f1 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 - x[0]*(-1) - x[1]*(-1)) # (-1, -1)
        f2 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 - x[0]*( 1) - x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 - x[0]*( 1) - x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 - x[0]*(-1) - x[1]*( 1)) # (-1,  1)
        f5 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*(-1)) # ( 0, -1)
        f6 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*( 1)) # ( 1,  0)
        f7 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*( 1)) # ( 0,  1)
        f8 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*(-1)) # (-1,  0)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]

        
    elif elem_type == 'TRI3':
        f1 = lambda x: x[0]
        f2 = lambda x: x[1]
        f3 = lambda x: 1 - x[0] - x[1]
        shape_fun = [f1, f2, f3]
    
    elif elem_type == 'TRI6':
        f1 = lambda x: x[0]*(2*x[0]-1)
        f2 = lambda x: x[1]*(2*x[1]-1)
        f3 = lambda x: (1-x[0]-x[1])*(1-2*x[0]-2*x[1])
        f4 = lambda x: 4*x[0]*x[1]
        f5 = lambda x: 4*x[1]*(1-x[0]-x[1])
        f6 = lambda x: 4*x[0]*(1-x[0]-x[1])
        shape_fun = [f1, f2, f3, f4, f5, f6]
            
    ############### 3D ###############
    elif elem_type == 'C3D8': # 3D linear element
    # elif elem_type == 38: 
        f1 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # (-1, 1, 1)
        f2 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # (-1,-1, 1)
        f3 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # (-1,-1,-1)
        f4 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # (-1, 1,-1)
        f5 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # ( 1, 1, 1)
        f6 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # ( 1,-1, 1)
        f7 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # ( 1,-1,-1)
        f8 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # ( 1, 1,-1)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]
         
    elif elem_type == 'D3BR20N': # 3D quadratic serendipity element
        # ref: http://what-when-how.com/the-finite-element-method/fem-for-3d-solids-finite-element-method-part-3/
        f1 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) * (x[0]*(-1)+x[1]*( 1)+x[2]*( 1)-2) # (-1, 1, 1)
        f2 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) * (x[0]*(-1)+x[1]*(-1)+x[2]*( 1)-2) # (-1,-1, 1)
        f3 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) * (x[0]*(-1)+x[1]*(-1)+x[2]*(-1)-2) # (-1,-1,-1)
        f4 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) * (x[0]*(-1)+x[1]*( 1)+x[2]*(-1)-2) # (-1, 1,-1)
        f5 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) * (x[0]*( 1)+x[1]*( 1)+x[2]*( 1)-2) # ( 1, 1, 1)
        f6 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) * (x[0]*( 1)+x[1]*(-1)+x[2]*( 1)-2) # ( 1,-1, 1)
        f7 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) * (x[0]*( 1)+x[1]*(-1)+x[2]*(-1)-2) # ( 1,-1,-1)
        f8 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) * (x[0]*( 1)+x[1]*( 1)+x[2]*(-1)-2) # ( 1, 1,-1)
        
        f9 =  lambda x: 1./4.*(1 - x[0]**2)*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # ( 0, 1, 1)
        f10 = lambda x: 1./4.*(1 - x[0]**2)*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # ( 0,-1, 1)
        f11 = lambda x: 1./4.*(1 - x[0]**2)*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # ( 0,-1,-1)
        f12 = lambda x: 1./4.*(1 - x[0]**2)*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # ( 0, 1,-1)
        
        f13 = lambda x: 1./4.*(1 - x[1]**2)*(1 + x[0]*( 1))*(1 + x[2]*( 1)) # ( 1, 0, 1)
        f14 = lambda x: 1./4.*(1 - x[1]**2)*(1 + x[0]*(-1))*(1 + x[2]*( 1)) # (-1, 0, 1)
        f15 = lambda x: 1./4.*(1 - x[1]**2)*(1 + x[0]*(-1))*(1 + x[2]*(-1)) # (-1, 0,-1)
        f16 = lambda x: 1./4.*(1 - x[1]**2)*(1 + x[0]*( 1))*(1 + x[2]*(-1)) # ( 1, 0,-1)
        
        f17 = lambda x: 1./4.*(1 - x[2]**2)*(1 + x[0]*( 1))*(1 + x[1]*( 1)) # ( 1, 1, 0)
        f18 = lambda x: 1./4.*(1 - x[2]**2)*(1 + x[0]*(-1))*(1 + x[1]*( 1)) # (-1, 1, 0)
        f19 = lambda x: 1./4.*(1 - x[2]**2)*(1 + x[0]*(-1))*(1 + x[1]*(-1)) # (-1,-1, 0)
        f20 = lambda x: 1./4.*(1 - x[2]**2)*(1 + x[0]*( 1))*(1 + x[1]*(-1)) # ( 1,-1, 0)
         
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]
    return shape_fun

def get_shape_grad_functions(elem_type):
    shape_fns = get_shape_val_functions(elem_type)
    return [jax.grad(f) for f in shape_fns]

@partial(jax.jit, static_argnames=['Gauss_num', 'dim', 'elem_type']) # necessary
def get_shape_vals(Gauss_num, dim, elem_type):
    """Pre-compute shape function values

    Returns
    -------
    shape_vals: ndarray
        (8, 8) = (quad_num, nodes_per_elem)  
    """
    shape_val_fns = get_shape_val_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_num, dim)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = np.array(shape_vals) # (quad_num, nodes_per_elem)
    # assert shape_vals.shape == (global_args['quad_num'], global_args['nodes_per_elem'])
    return shape_vals

@partial(jax.jit, static_argnames=['Gauss_num', 'dim', 'elem_type']) # necessary
def get_shape_grads(Gauss_num, dim, elem_type, XY, Elem_nodes):
    """Pre-compute shape function gradients

    Returns
    -------
    shape_grads_physical: ndarray
        (cell, quad_num, nodes_per_elem, dim)  
    JxW: ndarray
        (cell, quad_num)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_num, dim)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
            # Page 147, Eq. (3.9.3)
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
 
        shape_grads.append(physical_shape_grads)

    shape_grads = np.array(shape_grads) # (quad_num, nodes_per_elem, dim)
    # assert shape_grads.shape == (global_args['quad_num'], global_args['nodes_per_elem'], global_args['dim'])

    physical_nodal = np.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
    # physical_nodal: (nelem, none,      nodes_per_elem, dim, none)
    # shape_grads:   (none,      quad_num, nodes_per_elem, none, dim)
    # (nelem, quad_num, nodes_per_elem, dim, dim) -> (nelem, quad_num, 1, dim, dim)
    jacobian_dx_deta = np.sum(physical_nodal[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True)
    
    jacbian_det = np.squeeze(np.linalg.det(jacobian_dx_deta)) # (nelem, quad_num)
    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
    # print(jacobian_deta_dx[0, :, 0, :, :])
    # print(shape_grads)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    # print(shape_grads_physical[0])

    JxW = jacbian_det * quad_weights[None, :]
    return shape_grads_physical, JxW # (nelem, quad_num, nodes_per_elem, dim), (nelem, quad_num)


def get_adj_mat(Elem_nodes, nnode, s_patch):
    # Sparse matrix multiplication for graph theory.
    
    # get adjacency matrix of graph theory based on nodal connectivity
    adj_rows, adj_cols = [], []
    # self 
    for inode in range(nnode):
        adj_rows += [inode]
        adj_cols += [inode]
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        for (inode, jnode) in combinations(list(elem_nodes), 2):
            adj_rows += [inode, jnode]
            adj_cols += [jnode, inode]
    adj_values = onp.ones(len(adj_rows), dtype=onp.int32)
    adj_rows = onp.array(adj_rows, dtype=onp.int32)
    adj_cols = onp.array(adj_cols, dtype=onp.int32)
    
    # build sparse matrix
    adj_sp = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    adj_s = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    # print(adj_sp.toarray())

    # compute s th power of the adjacency matrix to get s th order of connectivity
    for itr in range(s_patch-1):
        adj_s = adj_s.dot(adj_sp)
    indices = adj_s.indices
    indptr = adj_s.indptr
    # print(adj_s.toarray())
    return indices, indptr


def get_dex_max(indices, indptr, s_patch, Elem_nodes, nelem, nodes_per_elem, dim): # delete d_c, XY, nnode
    
    edex_max = (2+2*s_patch)**dim # estimated value of edex_max
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (nelem, nodes_per_elem)
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
            nodal_patch_nodes0 = indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ] # global index, # node_idx 0
            nodal_patch_nodes1 = indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ] # global index
            ndexes[ielem, :] = onp.array([len(nodal_patch_nodes0),len(nodal_patch_nodes1)])
            elemental_patch_nodes = onp.unique(onp.concatenate((nodal_patch_nodes0, nodal_patch_nodes1)))  
            
        
        elif len(elem_nodes) == 4 and dim == 2: # 2D 4-node element
            nodal_patch_nodes0 = indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ] # global index, # node_idx 0
            nodal_patch_nodes1 = indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ] # global index
            nodal_patch_nodes2 = indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ] # global index
            nodal_patch_nodes3 = indices[ indptr[elem_nodes[3]] : indptr[elem_nodes[3]+1] ] # global index, # node_idx 3
            
            ndexes[ielem, :] = onp.array([len(nodal_patch_nodes0),len(nodal_patch_nodes1),
                                          len(nodal_patch_nodes2),len(nodal_patch_nodes3)])
            elemental_patch_nodes = onp.unique(onp.concatenate((nodal_patch_nodes0, nodal_patch_nodes1, 
                                                                nodal_patch_nodes2, nodal_patch_nodes3)))  # node_idx 3
            
            
        elif len(elem_nodes) == 8 and dim == 3: # 3D 8-node element
            nodal_patch_nodes0 = indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ] # global index, # node_idx 0
            nodal_patch_nodes1 = indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ] # global index
            nodal_patch_nodes2 = indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ] # global index
            nodal_patch_nodes3 = indices[ indptr[elem_nodes[3]] : indptr[elem_nodes[3]+1] ] # global index, # node_idx 3
            nodal_patch_nodes4 = indices[ indptr[elem_nodes[4]] : indptr[elem_nodes[4]+1] ] # global index, # node_idx 4
            nodal_patch_nodes5 = indices[ indptr[elem_nodes[5]] : indptr[elem_nodes[5]+1] ] # global index
            nodal_patch_nodes6 = indices[ indptr[elem_nodes[6]] : indptr[elem_nodes[6]+1] ] # global index
            nodal_patch_nodes7 = indices[ indptr[elem_nodes[7]] : indptr[elem_nodes[7]+1] ] # global index, # node_idx 7
            
            ndexes[ielem, :] = onp.array([len(nodal_patch_nodes0),len(nodal_patch_nodes1),
                                          len(nodal_patch_nodes2),len(nodal_patch_nodes3),
                                          len(nodal_patch_nodes4),len(nodal_patch_nodes5),
                                          len(nodal_patch_nodes6),len(nodal_patch_nodes7)])
            elemental_patch_nodes = onp.unique(onp.concatenate((nodal_patch_nodes0, nodal_patch_nodes1, 
                                                                nodal_patch_nodes2, nodal_patch_nodes3,
                                                                nodal_patch_nodes4, nodal_patch_nodes5, 
                                                                nodal_patch_nodes6, nodal_patch_nodes7)))  # node_idx 3
            
        
        edexes[ielem] = len(elemental_patch_nodes)
    edex_max = onp.max(edexes)
    ndex_max = onp.max(ndexes)
    return edex_max, ndex_max

def get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes, nelem, nodes_per_elem, dim): # for block, delete s_patch, d_c, XY
    
    # Assign memory to variables
    ## Elemental patch
    Elemental_patch_nodes_st = onp.zeros((nelem, edex_max), dtype=onp.int32) # edex_max should be grater than 100!
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    ## Nodal patch
    Nodal_patch_nodes_st = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_bool = onp.zeros((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_idx = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (nelem, nodes_per_elem, ndex_max)
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (nelem, nodes_per_elem)
    
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        
        # 1. for loop: nodal_patch_nodes in global nodal index
        for inode_idx, inode in enumerate(elem_nodes):
            nodal_patch_nodes = onp.sort(indices[ indptr[elem_nodes[inode_idx]] : indptr[elem_nodes[inode_idx]+1] ]) # global index
            ndex = len(nodal_patch_nodes)
            ndexes[ielem, inode_idx] = ndex
            Nodal_patch_nodes_st[ielem, inode_idx, :ndex] = nodal_patch_nodes  # global nodal index
            Nodal_patch_nodes_bool[ielem, inode_idx, :ndex] = onp.where(nodal_patch_nodes>=0, 1, 0)
        
        
        # 2. get elemental_patch_nodes    
        if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
            elemental_patch_nodes = onp.unique(onp.concatenate((Nodal_patch_nodes_st[ielem, 0, :ndexes[ielem, 0]],
                                                                Nodal_patch_nodes_st[ielem, 1, :ndexes[ielem, 1]])))  # node_idx 1
            
        elif len(elem_nodes) == 4 and dim == 2: # 2D 4-node element
            elemental_patch_nodes = onp.unique(onp.concatenate((Nodal_patch_nodes_st[ielem, 0, :ndexes[ielem, 0]],
                                                                Nodal_patch_nodes_st[ielem, 1, :ndexes[ielem, 1]],
                                                                Nodal_patch_nodes_st[ielem, 2, :ndexes[ielem, 2]],
                                                                Nodal_patch_nodes_st[ielem, 3, :ndexes[ielem, 3]])))  # node_idx 3
            
        elif len(elem_nodes) == 8 and dim == 3: # 3D 8-node element
            elemental_patch_nodes = onp.unique(onp.concatenate((Nodal_patch_nodes_st[ielem, 0, :ndexes[ielem, 0]],
                                                                Nodal_patch_nodes_st[ielem, 1, :ndexes[ielem, 1]],
                                                                Nodal_patch_nodes_st[ielem, 2, :ndexes[ielem, 2]],
                                                                Nodal_patch_nodes_st[ielem, 3, :ndexes[ielem, 3]],
                                                                Nodal_patch_nodes_st[ielem, 4, :ndexes[ielem, 4]],
                                                                Nodal_patch_nodes_st[ielem, 5, :ndexes[ielem, 5]],
                                                                Nodal_patch_nodes_st[ielem, 6, :ndexes[ielem, 6]],
                                                                Nodal_patch_nodes_st[ielem, 7, :ndexes[ielem, 7]])))  # node_idx 7
            
        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
        
        # 3. for loop: get nodal_patch_nodes_idx
        for inode_idx, inode in enumerate(elem_nodes):
            nodal_patch_nodes_idx = onp.searchsorted(
                elemental_patch_nodes, Nodal_patch_nodes_st[ielem, inode_idx, :ndexes[ielem, inode_idx]]) # local index
            Nodal_patch_nodes_idx[ielem, inode_idx, :ndexes[ielem, inode_idx]] = nodal_patch_nodes_idx
   
            
    # Convert everything to device array
    Elemental_patch_nodes_st = np.array(Elemental_patch_nodes_st)
    edexes = np.array(edexes)
    Nodal_patch_nodes_st = np.array(Nodal_patch_nodes_st)
    Nodal_patch_nodes_bool = np.array(Nodal_patch_nodes_bool)
    Nodal_patch_nodes_idx = np.array(Nodal_patch_nodes_idx)
    ndexes = np.array(ndexes)
    
    return Elemental_patch_nodes_st, edexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes


def in_range(xi, lb, ub):
    # lb: lower bound, floating number
    # ub: upper bound, floating number
    return np.heaviside(xi-lb,1) * np.heaviside(ub-xi, 0)

@jax.jit
def get_R_cubicSpline(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = ((2/3 - 4*zI**2 + 4*zI**3         ) * in_range(zI, 0.0, 0.5) +    # phi_i
                        (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * in_range(zI, 0.5, 1.0))
    return R
v_get_R_cubicSpline = jax.vmap(get_R_cubicSpline, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian1(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI)
    return R
v_get_R_gaussian1 = jax.vmap(get_R_gaussian1, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian2(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**2)
    return R
v_get_R_gaussian2 = jax.vmap(get_R_gaussian2, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian3(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**3)
    return R
v_get_R_gaussian3 = jax.vmap(get_R_gaussian3, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian4(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**4)
    return R
v_get_R_gaussian4 = jax.vmap(get_R_gaussian4, in_axes = (None,0,None))

@jax.jit
def get_R_gaussian5(xy, xvi, a_dil):
    zI = np.linalg.norm(xy - xvi)/a_dil
    R = np.exp(-zI**5)
    return R
v_get_R_gaussian5 = jax.vmap(get_R_gaussian5, in_axes = (None,0,None))

@partial(jax.jit, static_argnames=['ndex_max', 'mbasis', 'radial_basis', 'dim']) # This will slower the function
def Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis, radial_basis, dim):
    """ 
    --- Inputs ---
    # xy: point of interest (dim,)
    # xv: ndoal coordinates of patch nodes. ()
    # ndex: number of nodse in the nodal patch
    # ndex_max: max of ndex, precomputed value
    # nodal_patch_nodes_bool: boolean vector that tells ~~~
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    
    """
    
    RP = np.zeros(ndex_max + mbasis, dtype=np.double)
    
    if radial_basis == 'cubicSpline':
        RP = RP.at[:ndex_max].set(v_get_R_cubicSpline(xy, xv, a_dil) * nodal_patch_nodes_bool)
    if radial_basis == 'gaussian1':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian1(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian2':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian2(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian3':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian3(xy, xv, a_dil) * nodal_patch_nodes_bool)        
    if radial_basis == 'gaussian5':
        RP = RP.at[:ndex_max].set(v_get_R_gaussian3(xy, xv, a_dil) * nodal_patch_nodes_bool)        

    
    if dim == 2:    
        if mbasis > 0: # 1st
            RP = RP.at[ndex_max   : ndex_max+ 3].set(np.array([1                  ,     # N 1
                                                               xy[0]              ,     # N x
                                                                          xy[1]   ]))   # N y
            
        if mbasis > 3: # 2nd
            RP = RP.at[ndex_max+3].set(xy[0]    * xy[1]   )  # N
            RP = RP.at[ndex_max+4].set(xy[0]**2           )  # N
            RP = RP.at[ndex_max+5].set(           xy[1]**2)  # N    
            
        if mbasis > 6: # 3rd
            RP = RP.at[ndex_max+ 6: ndex_max+10].set(np.array([xy[0]**2 * xy[1]   ,     # N x^2y
                                                               xy[0]    * xy[1]**2,     # N xy^2
                                                               xy[0]**3           ,     # N x^3
                                                                          xy[1]**3]))   # N y^3        
            
        if mbasis > 10: # 4th
            RP = RP.at[ndex_max+10: ndex_max+15].set(np.array([xy[0]**2 * xy[1]**2,     # N x^2y^2
                                                               xy[0]**3 * xy[1]   ,     # N x^3y
                                                               xy[0]    * xy[1]**3,     # N xy^3
                                                               xy[0]**4           ,     # N x^4
                                                                           xy[1]**4]))  # N y^4        
        if mbasis > 15: # 5th
            RP = RP.at[ndex_max+15: ndex_max+20].set(np.array([xy[0]**3 * xy[1]**2,     # N x^3y^2
                                                               xy[0]**2 * xy[1]**3,     # N x^2y^3
                                                               xy[0]**4 * xy[1]   ,     # N x^4y
                                                               xy[0]    * xy[1]**4,     # N xy^4
                                                               xy[0]**5           ,     # N x^5
                                                                          xy[1]**5]))   # N y^5        
            
    if dim == 3:    
        if mbasis > 0: # 1st
            RP = RP.at[ndex_max   : ndex_max+ 4].set(np.array([1                             ,      # N 1                                                              
                                                               xy[0]                         ,      # N x 
                                                                          xy[1]              ,      # N y
                                                                                     xy[2]   ]))    # N z
            
        if mbasis > 4: # 2nd
            RP = RP.at[ndex_max+ 4: ndex_max+10].set(np.array([xy[0]    * xy[1]              ,      # N xy
                                                                          xy[1]    * xy[2]   ,      # N yz
                                                               xy[0]               * xy[2]   ,      # N zx
                                                               xy[0]**2                      ,      # N x^2
                                                                          xy[1]**2           ,      # N y^2
                                                                                     xy[2]**2]))    # N z^2
            
        if mbasis > 10: # 3rd
            RP = RP.at[ndex_max+10: ndex_max+20].set(np.array([xy[0]**2 * xy[1]              ,      # N x^2y
                                                               xy[0]    * xy[1]**2           ,      # N xy^2
                                                                          xy[1]**2 * xy[2]   ,      # N y^2z
                                                                          xy[1]    * xy[2]**2,      # N yz^2
                                                               xy[0]               * xy[2]**2,      # N z^2x
                                                               xy[0]**2            * xy[2]   ,      # N zx^2
                                                               xy[0]    * xy[1]    * xy[2]   ,      # N xyz
                                                               xy[0]**3                      ,      # N x^3
                                                                          xy[1]**3           ,      # N y^3
                                                                                     xy[2]**3]))    # N z^3        
            
        if mbasis > 20: # 4th
            RP = RP.at[ndex_max+ 20: ndex_max+35].set(np.array([xy[0]**2 * xy[1]**2           ,     # N x^2y^2
                                                                           xy[1]**2 * xy[2]**2,     # N y^2z^2
                                                                xy[0]**2            * xy[2]**2,     # N z^2x^2
                                                                xy[0]**3 * xy[1]              ,     # N x^3y
                                                                xy[0]    * xy[1]**3           ,     # N xy^3    
                                                                           xy[1]**3 * xy[2]   ,     # N y^3z
                                                                           xy[1]    * xy[2]**3,     # N yz^3
                                                                xy[0]               * xy[2]**3,     # N z^3x
                                                                xy[0]**3            * xy[2]   ,     # N zx^3
                                                                xy[0]**2 * xy[1]    * xy[2]   ,     # N x^2yz
                                                                xy[0]    * xy[1]**2 * xy[2]   ,     # N xy^2z
                                                                xy[0]    * xy[1]    * xy[2]**2,     # N xyz^2
                                                                xy[0]**4                      ,     # N x^4
                                                                           xy[1]**4           ,     # N y^4
                                                                                      xy[2]**4]))   # N z^4        
            
        
    return RP

v_Compute_RadialBasis_1D = jax.vmap(Compute_RadialBasis_1D, in_axes = (0,None,None,None,None,
                                                                                   None,None,None,None), out_axes=1)
Compute_RadialBssis_der = jax.jacfwd(Compute_RadialBasis_1D, argnums=0)

@partial(jax.jit, static_argnames=['ndex_max','a_dil','mbasis','radial_basis','dim']) # unneccessary
def get_G(ndex, nodal_patch_nodes, nodal_patch_nodes_bool, XY, ndex_max, a_dil, mbasis, radial_basis, dim):
    # nodal_patch_nodes_bool: (ndex_max,)
    G = np.zeros((ndex_max + mbasis, ndex_max + mbasis), dtype=np.double)
    xv = XY[nodal_patch_nodes,:]
    RPs = v_Compute_RadialBasis_1D(xv, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                                  a_dil, mbasis, radial_basis, dim) # (ndex_max + mbasis, ndex_max)
    
    G = G.at[:,:ndex_max].set(RPs * nodal_patch_nodes_bool[None,:])                        
    
    # Make symmetric matrix
    G = np.tril(G) + np.triu(G.T, 1)
    
    # Build diagonal terms to nullify dimensions
    Imat = np.eye(ndex_max) * np.abs(nodal_patch_nodes_bool-1)[:,None]
    G = G.at[:ndex_max,:ndex_max].add(Imat)
    return G # G matrix

vv_get_G = jax.vmap(jax.vmap(get_G, in_axes = (0,0,0,None,None,  None,None,None,None)), in_axes = (0,0,0,None,None,  None,None,None,None))
#Gs: (num_cells, num_nodes, ndex_max+mbasis, ndex_max+mbasis)


@partial(jax.jit, static_argnames=['ndex_max','edex_max','a_dil','mbasis','radial_basis','dim']) # must
def get_Phi(G, nodal_patch_nodes, nodal_patch_nodes_bool, nodal_patch_nodes_idx, ndex, shape_val, elem_nodes,
            XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim): # 15
    
    xy_elem = XY[elem_nodes,:] # (nodes_per_elem, dim)
    xv = XY[nodal_patch_nodes,:]
    
    xy = np.sum(shape_val[:, None] * xy_elem, axis=0, keepdims=False)
    RP_1D = Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim) # (edex_max,) but only 'ndex+1' nonzero terms
    RP_der = Compute_RadialBssis_der(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim)
    RP = np.column_stack((RP_1D, RP_der))
    
    ## Standard matrix solver
    phi_org = np.linalg.solve(G.T, RP)[:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    
    ## when the moment matrix is singular
    # phi_lstsq = np.linalg.lstsq(G.T, RP)
    # phi_org = phi_lstsq[0][:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    
    ## Assemble
    phi = np.zeros((edex_max + 1, 1+dim))  # trick, add dummy node at the end
    phi = phi.at[nodal_patch_nodes_idx, :].set(phi_org) 
    phi = phi[:edex_max,:] # trick, delete dummy node
    
    return phi

vvv_get_Phi = jax.vmap(jax.vmap(jax.vmap(get_Phi, in_axes = (0,0,0,0,0,None,None,None,None,None,None,None,None,None)), 
                                in_axes = (None,None,None,None,None,0,None,None,None,None,None,None,None,None)),
                                in_axes = (0,0,0,0,0,None,0,None,None,None,None,None,None,None))

vvv_get_Phi_face = jax.vmap(jax.vmap(jax.vmap(get_Phi, in_axes = (0,0,0,0,0,None,None,None,None,None,None,None,None,None)), 
                                in_axes = (None,None,None,None,None,0,None,None,None,None,None,None,None,None)),
                                in_axes = (0,0,0,0,0,0,0,None,None,None,None,None,None,None))

@partial(jax.jit, static_argnames=['ndex_max','edex_max','a_dil','mbasis','radial_basis','dim']) # must
def get_Phi_singular(G, nodal_patch_nodes, nodal_patch_nodes_bool, nodal_patch_nodes_idx, ndex, shape_val, elem_nodes,
            XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim): # 7
    
    xy_elem = XY[elem_nodes,:] # (nodes_per_elem, dim)
    xv = XY[nodal_patch_nodes,:]
    
    xy = np.sum(shape_val[:, None] * xy_elem, axis=0, keepdims=False)
    RP_1D = Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim) # (edex_max,) but only 'ndex+1' nonzero terms
    RP_der = Compute_RadialBssis_der(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis, radial_basis, dim)
    RP = np.column_stack((RP_1D, RP_der))
    
    ## Standard matrix solver
    # phi_org = np.linalg.solve(G.T, RP)[:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    
    ## when the moment matrix is singular
    phi_lstsq = np.linalg.lstsq(G.T, RP)
    phi_org = phi_lstsq[0][:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    
    ## Assemble
    phi = np.zeros((edex_max + 1, 1+dim))  # trick, add dummy node at the end
    phi = phi.at[nodal_patch_nodes_idx, :].set(phi_org) 
    phi = phi[:edex_max,:] # trick, delete dummy node
    
    return phi

vvv_get_Phi_singular = jax.vmap(jax.vmap(jax.vmap(get_Phi_singular, in_axes = (0,0,0,0,0,  None,None,None,None,None,  None,None,None,None)), 
                                in_axes = (None,None,None,None,None,  0,None,None,None,None,  None,None,None,None)),
                                in_axes = (0,0,0,0,0,  None,0,None,None,None,  None,None,None,None))

def get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                       XY, XY_host, Elem_nodes_block, Elem_nodes_block_host, shape_vals, shape_grads_physical_block, JxW_block,
                       Gauss_num_CFEM, quad_num_CFEM, dim, nodes_per_elem,
                       indices, indptr, s_patch, edex_max, ndex_max, a_dil, mbasis, radial_basis, bool_singular_check = True):
    
    
    # start_init = time.time()
    elem_idx_block_host = onp.array(elem_idx_block) # cpu # (num_cells)
    elem_idx_block = np.array(elem_idx_block) # gpu
    Elem_nodes_block_host = Elem_nodes_block_host[elem_idx_block_host] # cpu
    Elem_nodes_block = Elem_nodes_block[elem_idx_block] # gpu
    
    # start_patch = time.time()
    (Elemental_patch_nodes_st_block, edexes_block,
     Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block
     ) = get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes_block_host, nelem_per_block, nodes_per_elem, dim)                            
    
    # start_G = time.time()
    Gs = vv_get_G(ndexes_block, Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, XY, ndex_max, a_dil, mbasis, radial_basis, dim)
    # Gs: (num_cells, num_nodes, ndex_max+mbasis, ndex_max+mbasis)
    # print('G', time.time() - start_G)
        
    ############# Phi ###############
    # start_Phi = time.time()
    Phi = vvv_get_Phi(Gs, Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block, shape_vals, Elem_nodes_block,
                XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim) # [nelem_per_block, quad_num, nodes_per_elem, edex_max, 1+dim]
    #print('shape_vals',shape_vals)
    ## Deals with singular moment matrices
    if np.any(np.isnan(Phi)) and bool_singular_check: # If Phi has 'nan' components, i compute them again with serial mode.
        nan_indices = np.argwhere(np.isnan(Phi))
        idx0 = nan_indices[:,0] # elements
        idx0_unique = np.unique(idx0)
        # print(f"[Warning] {elem_idx_block[0]//nelem_per_block} block {idx0_unique} elem have 'nan' components on Phi")
        print(f"[Warning] {elem_idx_block[0]//nelem_per_block} block have 'nan' components on Phi")
        
        Gs_sing = Gs[idx0_unique, :, :,:] # argnum 0
        Nodal_patch_nodes_st_block_sing = Nodal_patch_nodes_st_block[idx0_unique,:,:] # argnum 1
        Nodal_patch_nodes_bool_block_sing = Nodal_patch_nodes_bool_block[idx0_unique,:,:] # argnum 2
        Nodal_patch_nodes_idx_block_sing = Nodal_patch_nodes_idx_block[idx0_unique,:,:] # argnum 3
        ndexes_block_sing = ndexes_block[idx0_unique,:] # argnum 4
        Elem_nodes_block_sing = Elem_nodes_block[idx0_unique,:] # argnum 6
        
        Phi_singular = vvv_get_Phi_singular(Gs_sing, 
                    Nodal_patch_nodes_st_block_sing, Nodal_patch_nodes_bool_block_sing, Nodal_patch_nodes_idx_block_sing, ndexes_block_sing, 
                    shape_vals, Elem_nodes_block_sing,
                    XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim)
        Phi = Phi.at[idx0_unique,:,:,:,:].set(Phi_singular) 
    
    # start_PoU = time.time()
    N_til_block = np.sum(shape_vals[None, :, :, None]*Phi[:,:,:,:,0], axis=2) # (num_cells, num_quads, edex_max)
    if bool_singular_check: # Default True. Becomes False when we plot mesh because we do not need gradients.
        Grad_N_til_block = (np.sum(shape_grads_physical_block[:, :, :, None, :]*Phi[:,:,:,:,:1], axis=2) 
                      + np.sum(shape_vals[None, :, :, None, None]*Phi[:,:,:,:,1:], axis=2) )
    
        # Check partition of unity
        if not ( np.allclose(np.sum(N_til_block, axis=2), np.ones((nelem_per_block, quad_num_CFEM), dtype=np.double)) and
                np.allclose(np.sum(Grad_N_til_block, axis=2), np.zeros((nelem_per_block, quad_num_CFEM, dim), dtype=np.double)) ):
            
            # print(np.sum(N_til_block, axis=2))
            PoU_Check_N = (np.linalg.norm(np.sum(N_til_block, axis=2) - np.ones((nelem_per_block, quad_num_CFEM), dtype=np.float64))**2/(nelem_per_block*quad_num_CFEM))**0.5
            PoU_Check_Grad_N = (np.linalg.norm(np.sum(Grad_N_til_block, axis=2))**2/(nelem_per_block*quad_num_CFEM*dim))**0.5
            if PoU_Check_Grad_N >= 1e-6 or np.isnan(PoU_Check_Grad_N):
                print(f"PoU Check failed at element {elem_idx_block[0]}~{elem_idx_block[-1]}")
                print(f'PoU check N / Grad_N: {PoU_Check_N:.4e} / {PoU_Check_Grad_N:.4e}')
    else:
        Grad_N_til_block = 0
    # print('PoU', time.time() - start_PoU)
        
    return N_til_block, Grad_N_til_block, Elemental_patch_nodes_st_block

def get_CFEM_shape_fun_face(elem_idx_block, nelem_per_block,
                       XY, Elem_nodes_face_full, shape_vals,
                       dim, nodes_per_elem,
                       indices, indptr, edex_max, ndex_max, a_dil, mbasis, radial_basis, bool_singular_check = True):
    
    #elem_idx_block from core program: 根据内存限制的for loop而来
    elem_idx_block = np.array(elem_idx_block) # gpu
    
    #Elem_nodes_face_full: Nodes of Face elements
    Elem_nodes_block = Elem_nodes_face_full[elem_idx_block]  # full nodes of the elements on the face
    Elem_nodes_block_host = onp.array(Elem_nodes_block) # full nodes for patch function

    # start_patch = time.time()
    (Elemental_patch_nodes_st_block, edexes_block,
     Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block
     ) = get_patch_info(indices, indptr, edex_max, ndex_max, Elem_nodes_block_host, nelem_per_block, nodes_per_elem, dim)                            
    
    # print('Elemental_patch_nodes_st_block',Elemental_patch_nodes_st_block)
    # print('Nodal_patch_nodes_st_block',Nodal_patch_nodes_st_block)
    #shape_vals [num_quad, num_node]
    Gs = vv_get_G(ndexes_block, Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, XY, ndex_max, a_dil, mbasis, radial_basis, dim)
    #print('Gs',Gs)
    # Gs: dim (num_ele, ...)
    Phi = vvv_get_Phi_face(Gs, Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block, shape_vals, Elem_nodes_block,
                XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim) # [nelem_per_block, quad_num, nodes_per_elem, edex_max, 1+dim]
    #print('shape_vals',shape_vals)
    #print('Phi',Phi)
    if np.any(np.isnan(Phi)) and bool_singular_check: # If Phi has 'nan' components, i compute them again with serial mode.
        nan_indices = np.argwhere(np.isnan(Phi))
        idx0 = nan_indices[:,0] # elements
        idx0_unique = np.unique(idx0)
        # print(f"[Warning] {elem_idx_block[0]//nelem_per_block} block {idx0_unique} elem have 'nan' components on Phi")
        print(f"[Warning] {elem_idx_block[0]//nelem_per_block} block have 'nan' components on Phi")
        
        Gs_sing = Gs[idx0_unique, :, :,:] # argnum 0
        Nodal_patch_nodes_st_block_sing = Nodal_patch_nodes_st_block[idx0_unique,:,:] # argnum 1
        Nodal_patch_nodes_bool_block_sing = Nodal_patch_nodes_bool_block[idx0_unique,:,:] # argnum 2
        Nodal_patch_nodes_idx_block_sing = Nodal_patch_nodes_idx_block[idx0_unique,:,:] # argnum 3
        ndexes_block_sing = ndexes_block[idx0_unique,:] # argnum 4
        Elem_nodes_block_sing = Elem_nodes_block[idx0_unique,:] # argnum 6
        
        Phi_singular = vvv_get_Phi_singular(Gs_sing, 
                    Nodal_patch_nodes_st_block_sing, Nodal_patch_nodes_bool_block_sing, Nodal_patch_nodes_idx_block_sing, ndexes_block_sing, 
                    shape_vals, Elem_nodes_block_sing,
                    XY, ndex_max, edex_max, a_dil, mbasis, radial_basis, dim)
        Phi = Phi.at[idx0_unique,:,:,:,:].set(Phi_singular) 

    N_til_face = np.sum(shape_vals[:, :, :, None]*Phi[:, :, :, :, 0], axis=2) # (num_cells, num_quads, edex_max) #need nanson's formula later
        
    return N_til_face, Elemental_patch_nodes_st_block


def get_connectivity(Elemental_patch_nodes, dim):
    # get connectivity vector for 2D element
    (nelem, edex_max) = Elemental_patch_nodes.shape
    connectivity = np.zeros((nelem, edex_max*dim), dtype = np.int64)
    connectivity = connectivity.at[:, np.arange(0,edex_max*dim, dim)].set(Elemental_patch_nodes*dim)
    connectivity = connectivity.at[:, np.arange(1,edex_max*dim, dim)].set(Elemental_patch_nodes*dim+1)
    if dim == 3:
        connectivity = connectivity.at[:, np.arange(2,edex_max*dim, dim)].set(Elemental_patch_nodes*dim+2)
    return connectivity

def get_elements(ele_type):
    """Mesh node ordering is important.
    If the input mesh file is Gmsh .msh or Abaqus .inp, meshio would convert it to
    its own ordering. My experience shows that meshio ordering is the same as Abaqus.
    For example, for a 10-node tetrahedron element, the ordering of meshio is the following
    https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html
    The troublesome thing is that basix has a different ordering. As shown below
    https://defelement.com/elements/lagrange.html
    The consequence is that we need to define this "re_order" variable to make sure the 
    ordering is correct.
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 6 # 2x2x2, TODO: is this full integration?   to make for the surface integral also has 4 points for 1 direction
        degree = 1
    elif ele_type == 'HEX27':
        print(f"Warning: 27-node hexahedron is rarely used in practice and not recommended.")
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 10 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
        degree = 2
    # TODO: Check if this is correct.
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 
        degree = 2 
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0 # 1, full integration    
        degree = 1    
    elif  ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 # 3, full integration 
        degree = 2 
    else:
        raise NotImplementedError

    return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order


def reorder_inds(inds, re_order):
    new_inds = []
    for ind in inds.reshape(-1): 
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type):
    """TODO: Add comments

    Returns
    ------- 
    shape_values: ndarray @ gauss point 
        (8, 8) = (num_quads, num_nodes)
    shape_grads_ref: ndarray @ gauss point
        (8, 8, 3) = (num_quads, num_nodes, dim)
    weights: ndarray
        (8,) = (num_quads,)
    """
    element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements(ele_type)
    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)  
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    print(f"ele_type = {ele_type}, quad_points.shape = {quad_points.shape}")
    return shape_values, shape_grads_ref, weights



def get_face_shape_vals_and_grads(ele_type):
    """TODO: Add comments
    What is the face order?
    Returns
    ------- 
    face_shape_vals: ndarray
        (6, 4, 8) = (num_faces, num_face_quads, num_nodes)
    face_shape_grads_ref: ndarray
        (6, 4, 3) = (num_faces, num_face_quads, num_nodes, dim)
    face_weights: ndarray
        (6, 4) = (num_faces, num_face_quads)
    face_normals:ndarray
        (6, 3) = (num_faces, dim)
    face_inds: ndarray
        (6, 4) = (num_faces, num_face_vertices)
    """
    element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order = get_elements(ele_type)

    # TODO: Check if this is correct.
    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1] 
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)

    
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    print(f"face_quad_points.shape = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds

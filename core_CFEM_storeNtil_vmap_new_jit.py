import numpy as onp
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO, BCSR
import scipy
import sys
import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union

from generate_mesh import Mesh
from basis_CFEM import *
from GPU_support import *
import os
from jax.config import config


config.update("jax_enable_x64", True)


onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


@dataclass
class FEM:
    """
    Solving second-order elliptic PDE problems whose FEM weak form is 
    (f(u_grad), v_grad) * dx - (traction, v) * ds - (body_force, v) * dx = 0,
    where u and v are trial and test functions, respectively, and f is a general function.
    This covers
        - Poisson's problem
        - Heat equation
        - Linear elasticity
        - Hyper-elasticity #large deformation?
        - Plasticity
        ...

    Attributes
    ----------
    mesh : Mesh object
        The mesh object stores points (coordinates) and cells (connectivity).
    vec : int
        The number of vector variable components of the solution.
        E.g., a 3D displacement field has u_x, u_y and u_z components, so vec=3 
    dim : int
        The dimension of the problem.
    ele_type : str
        Element type
    dirichlet_bc_info : [location_fns, vecs, value_fns]
        location_fns : List[Callable]
            Callable : a function that inputs a point and returns if the point satisfies the location condition
        vecs: List[int]
            integer value must be in the range of 0 to vec - 1, 
            specifying which component of the (vector) variable to apply Dirichlet condition to
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Dirichlet value
    periodic_bc_info : [location_fns_A, location_fns_B, mappings, vecs]
        location_fns_A : List[Callable]
            Callable : location function for boundary A
        location_fns_B : List[Callable]
            Callable : location function for boundary B
        mappings : List[Callable]
            Callable: function mapping a point from boundary A to boundary B
        vecs: List[int]
            which component of the (vector) variable to apply periodic condition to 
    neumann_bc_info : [location_fns, value_fns]
        location_fns : List[Callable]
            Callable : location function for Neumann boundary
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Neumann value
    cauchy_bc_info : [location_fns, value_fns]
        location_fns : List[Callable]
            Callable : location function for Cauchy boundary
        value_fns : List[Callable]
            Callable : a function that inputs the solution and returns the Cauchy boundary value
    source_info: Callable
        A function that inputs a point and returns the body force at this point
    additional_info : Any
        Other information that the FEM solver should know
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    dirichlet_bc_info: Optional[List[Union[List[Callable], List[int], List[Callable]]]] = None 
    periodic_bc_info: Optional[List[Union[List[Callable], List[Callable], List[Callable], List[int]]]] = None
    neumann_bc_info: Optional[List[Union[List[Callable], List[Callable]]]] = None
    cauchy_bc_info: Optional[List[Union[List[Callable], List[Callable]]]] = None
    source_info: Callable = None
    additional_info: Any = ()

    def __post_init__(self):
        
        self.points = self.mesh.points #(num_total_nodes, dim): nodal coord for each node
        self.points_device = np.array(self.points)
        self.cells = self.mesh.cells  #(num_cell, num_nodes): nodal index for each element
        self.cells_device = np.array(self.cells)
        
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes*self.vec
        self.internal_vars = {}
        self.custom_init(*self.additional_info)


        self.radial_basis = 'cubicSpline' 
        # radial_basis = 'gaussian4'


        start_time = time.time()
        print(f"Start timing - Compute shape function values, gradients, etc.")

        #(num_quads, num_nodes)
        self.shape_vals = get_shape_vals(self.Gauss_Num_CFEM, self.dim, self.ele_type_CFEM)
        #self.shape_vals_face = get_shape_vals(self.Gauss_Num_CFEM, self.dim - 1, self.ele_type_CFEM)
        #print('shape_vals_face', self.shape_vals_face)
        
        #dim physical shape grads (num_cells, num_quads, num_nodes, dim) ; Jacobian(num_cells, num_quads)
        self.shape_grads, self.JxW = get_shape_grads(self.Gauss_Num_CFEM, self.dim, self.ele_type_CFEM, self.points, self.cells)

        self.num_nodes = self.shape_vals.shape[1]
        
        self.ele_dof = self.num_nodes * self.dim
        
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds = get_face_shape_vals_and_grads(self.ele_type)
        
        self.face_ele, self.selected_face_shape_vals_collection = self.get_face_ele_selected_face_shape_vals()  #get the all the elements with neumann bc face nodes 

        iblock = 0
        mem_report(iblock, self.gpu_idx)
        self.indices, self.indptr = get_adj_mat(self.cells, self.num_total_nodes, self.s_patch)

        self.p_dict={0:0, 1:4, 2:10, 3:20, 4:35}
        self.mbasis = self.p_dict[self.poly]
        self.d_c = self.L / self.nelem_y     # characteristic length in physical coord.
        # print('dc',d_c)
        self.a_dil = self.alpha_dils * self.d_c      
        self.edex_max, self.ndex_max = get_dex_max(self.indices, self.indptr, self.s_patch, self.cells, self.num_cells, self.num_nodes, self.dim)

        
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(self.dirichlet_bc_info)
        self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_boundary_conditions()
        self.body_force = self.compute_body_force_by_fn()

        print(f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs.")

        
        self.Grad_N_til_blocks, self.JxW_blocks, self.v_grads_JxW_blocks, self.Elemental_patch_nodes_st_blocks, self.connectivity_blocks = self.get_blocks()
        # for dirichlet b.c., no need to execute the following lines:
        if not self.face_ele == []:
            self.face_ele = np.array(self.face_ele).reshape(-1, self.num_nodes)        
            self.num_faces = self.face_ele.shape[0]
            print('number of faces:', self.num_faces)
            self.N_til_faces, self.face_connectivity_blocks = self.get_face_blocks()
        self.pre_jit_fns()
        compute_time = time.time() - start_time 
        print(f"Done pre-computations of all blocks, took {compute_time} [s]")

    # end of modifying for Ntilda and Btilda
    def custom_init(self):
        """Child class should override if more things need to be done in initialization
        """
        pass
    
    def get_face_ele_selected_face_shape_vals(self,  **internal_vars):
        """
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, j] returns the index of face j of cell i of surface k; i cell number; j face number
        """
        self.compute_Neumann_boundary_inds()
        face_cells = []
        selected_face_shape_vals_collection = []
        if self.neumann_bc_info is not None:
            for i, boundary_inds in enumerate(self.neumann_boundary_inds_list):  #self.neumann_boundary_inds_list: this is the faceID
                if 'neumann_vars' in internal_vars.keys():
                    int_vars = internal_vars['neumann_vars'][i]
                else:
                    int_vars = ()
                # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
                subset_quad_points = self.get_physical_surface_quad_points(boundary_inds) 
                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                subset_cells = np.take(self.cells, boundary_inds[:, 0], axis=0) # (num_selected_faces, num_nodes)
                face_cells.append(subset_cells)
                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                selected_face_shape_vals = np.take(self.face_shape_vals, boundary_inds[:, 1], axis=0)
                num_face_quads = selected_face_shape_vals.shape[1]
                selected_face_shape_vals_collection.append(selected_face_shape_vals)
            selected_face_shape_vals_collection = np.array(selected_face_shape_vals_collection).reshape(-1, num_face_quads, self.num_nodes)
        return face_cells, selected_face_shape_vals_collection
    
    def get_nblock(self):
        max_array_size_block = 1e9 #8  # 2e9 for Athena
        max_GPU_memory = 10*1000 # MB
        size_Phi = int(self.num_cells) * int(self.num_quads) * int(self.ele_dof) * int(self.edex_max) * int(1 + self.dim)
        size_Gs  = int(self.num_cells) * int(self.num_quads) * int(self.ele_dof) * int((self.ndex_max + self.mbasis)**2)
        self.nblock = int(max(size_Phi, size_Gs) // max_array_size_block + 1) # regular blocks + 1 remainder
        
        self.nelem_per_block_regular = self.num_cells // self.nblock # set any value
        (quo, rem) = divmod(self.num_cells, self.nelem_per_block_regular)
        if rem == 0:
            self.nblock = quo
            self.nelem_per_block_remainder = self.nelem_per_block_regular
        else:
            self.nblock = quo + 1
            self.nelem_per_block_remainder = rem            
        print(f"CFEM A_sp -> {self.nblock} blocks with {self.nelem_per_block_regular} elems per block and remainder {self.nelem_per_block_remainder}")
        assert self.num_cells == (self.nblock-1) * self.nelem_per_block_regular + self.nelem_per_block_remainder
    
    def get_nblock_face(self):
        max_array_size_block = 1e9 #8  # 2e9 for Athena
        max_GPU_memory = 10*1000 # MB
        size_Phi = int(self.num_faces) * int(self.num_quads) * int(self.ele_dof) * int(self.edex_max) * int(1 + self.dim)
        size_Gs  = int(self.num_faces) * int(self.num_quads) * int(self.ele_dof) * int((self.ndex_max + self.mbasis)**2)
        self.nblock_face = int(max(size_Phi, size_Gs) // max_array_size_block + 1) # regular blocks + 1 remainder
        print('nblock_face', self.nblock_face)
        self.nelem_per_block_regular_face = self.num_faces // self.nblock_face # set any value
        print('nelem_per_block_regular_face',self.nelem_per_block_regular_face)
        (quo, rem) = divmod(self.num_faces, self.nelem_per_block_regular_face)
        if rem == 0:
            self.nblock_face = quo
            self.nelem_per_block_remainder_face = self.nelem_per_block_regular_face
        else:
            self.nblock_face = quo + 1
            self.nelem_per_block_remainder_face = rem  
        print(f"CFEM A_sp face -> {self.nblock_face} blocks with {self.nelem_per_block_regular_face} elems per block and remainder {self.nelem_per_block_remainder_face}")          
    

    
    def get_blocks(self):
        start = time.time()
        self.get_nblock()
        nelem_per_block = self.nelem_per_block_regular
        elem_idx_block = np.array(range(0, nelem_per_block), dtype=np.int32)

        
        #N_til dim:(Num_cells_in_block, Num_quads, Edex_max)
        #Grad_N_til dim:(Num_cells_in_block, Num_quads, Edex_max, dim)
        #JxW (Num_cells_in_block, Num_quads) self.JxW
        
        ( N_til_blocks , Grad_N_til_blocks,
        Elemental_patch_nodes_st_blocks) = get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                self.points_device, self.points, self.cells_device, self.cells, self.shape_vals, self.shape_grads[elem_idx_block, :,:,:], self.JxW,
                self.Gauss_Num_CFEM, self.quad_num_CFEM, self.dim, self.num_nodes, 
                self.indices, self.indptr, self.s_patch, self.edex_max, self.ndex_max, self.a_dil, self.mbasis, self.radial_basis)

        connectivity_blocks = get_connectivity(Elemental_patch_nodes_st_blocks, self.dim)
        v_grads_JxW_blocks = Grad_N_til_blocks[:, :, :, None, :] * self.JxW[elem_idx_block, :, None, None, None]

        for iblock in range(1, self.nblock):
            if iblock % 50 ==0:
                print('Now running block', iblock)
            if iblock == self.nblock-1:
                nelem_per_block = self.nelem_per_block_remainder
                elem_idx_block = np.array(range(self.nelem_per_block_regular*iblock, self.nelem_per_block_regular*iblock 
                                                + self.nelem_per_block_remainder), dtype=np.int32)
            else:
                nelem_per_block = self.nelem_per_block_regular
                elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)


            ( N_til_block , Grad_N_til_block,
            Elemental_patch_nodes_st_block) = get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                    self.points_device, self.points, self.cells_device, self.cells, self.shape_vals, self.shape_grads[elem_idx_block, :,:,:], self.JxW,
                    self.Gauss_Num_CFEM, self.quad_num_CFEM, self.dim, self.num_nodes, 
                    self.indices, self.indptr, self.s_patch, self.edex_max, self.ndex_max, self.a_dil, self.mbasis, self.radial_basis)

            connectivity_block = get_connectivity(Elemental_patch_nodes_st_block, self.dim)
            v_grads_JxW_block = Grad_N_til_block[:, :, :, None, :] * self.JxW[elem_idx_block, :, None, None, None]

            N_til_blocks = onp.concatenate((N_til_blocks, N_til_block))
            Grad_N_til_blocks = onp.concatenate((Grad_N_til_blocks, Grad_N_til_block))
            v_grads_JxW_blocks = onp.concatenate((v_grads_JxW_blocks, v_grads_JxW_block))
            Elemental_patch_nodes_st_blocks = onp.concatenate((Elemental_patch_nodes_st_blocks, Elemental_patch_nodes_st_block))
            connectivity_blocks = onp.concatenate((connectivity_blocks, connectivity_block))
            
            #Grad_N_til dim:(Num_cells, Num_quads, Edex_max, dim)
            #JxW (Num_cells, Num_quads)

        return  (Grad_N_til_blocks, self.JxW, v_grads_JxW_blocks, Elemental_patch_nodes_st_blocks, connectivity_blocks)


    def get_face_blocks(self):
        start = time.time()
        self.get_nblock_face()
        nface_per_block = self.nelem_per_block_regular_face
        face_idx_block = np.array(range(0, nface_per_block), dtype=np.int32)

        
        (N_til_faces, Elemental_patch_nodes_st_blocks) = get_CFEM_shape_fun_face(face_idx_block, nface_per_block,
                self.points_device, self.face_ele, self.selected_face_shape_vals_collection[face_idx_block, :,:], self.dim, self.num_nodes, 
                self.indices, self.indptr, self.edex_max, self.ndex_max, self.a_dil, self.mbasis, self.radial_basis)
 
        connectivity_blocks = get_connectivity(Elemental_patch_nodes_st_blocks, self.dim)

        for iblock in range(1, self.nblock_face):
            if iblock % 50 ==0:
                print('Now running block', iblock)
            if iblock == self.nblock_face-1:
                nface_per_block = self.nelem_per_block_remainder_face
                face_idx_block = np.array(range(self.nelem_per_block_regular_face*iblock, self.nelem_per_block_regular_face*iblock 
                                                + self.nelem_per_block_remainder_face), dtype=np.int32)
            else:
                nface_per_block = self.nelem_per_block_regular_face
                face_idx_block = np.array(range(nface_per_block*iblock, nface_per_block*(iblock+1)), dtype=np.int32)


            # (N_til_face, Elemental_patch_nodes_st_block) = get_CFEM_shape_fun_face(elem_idx_block, nelem_per_block,
            #         self.points_device, self.points, self.cells_device, self.cells, self.selected_face_shape_vals_collection, self.shape_grads[elem_idx_block, :,:,:], self.JxW,
            #         self.Gauss_Num_CFEM, self.quad_num_CFEM, self.dim, self.num_nodes, 
            #         self.indices, self.indptr, self.s_patch, self.edex_max, self.ndex_max, self.a_dil, self.mbasis, self.radial_basis)

            (N_til_face, Elemental_patch_nodes_st_block) = get_CFEM_shape_fun_face(face_idx_block, nface_per_block,
                    self.points_device, self.face_ele, self.selected_face_shape_vals_collection[face_idx_block, :,:], self.dim, self.num_nodes, 
                    self.indices, self.indptr, self.edex_max, self.ndex_max, self.a_dil, self.mbasis, self.radial_basis)

            connectivity_block = get_connectivity(Elemental_patch_nodes_st_block, self.dim)
            N_til_faces = onp.concatenate((N_til_faces, N_til_face))
            Elemental_patch_nodes_st_blocks = onp.concatenate((Elemental_patch_nodes_st_blocks, Elemental_patch_nodes_st_block))
            connectivity_blocks = onp.concatenate((connectivity_blocks, connectivity_block))

        return  (N_til_faces, connectivity_blocks)


    def get_face_shape_grads(self, boundary_inds):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change
        
        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            (num_selected_faces, 2)
        
        Returns
        -------
        face_shape_grads_physical : onp.ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : onp.ndarray
            (num_selected_faces, num_face_quads)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]] # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = onp.linalg.det(jacobian_dx_deta) # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta) # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights # (num_selected_faces, num_face_quads)
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self):
        """Compute physical quadrature points
        Prepare to compute body force
        Returns
        -------
        physical_quad_points : onp.ndarray
            (num_cells, num_quads, dim) 
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim) 
        physical_quad_points = onp.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return physical_quad_points

    def get_physical_surface_quad_points(self, boundary_inds):
        """Compute physical quadrature points on the surface
        Prepare for compute neumann bc
        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            ndarray shape: (num_selected_faces, 2)
        
        Returns
        -------
        physical_surface_quad_points : ndarray
            (num_selected_faces, num_face_quads, dim) 
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)  
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim) 
        physical_surface_quad_points = onp.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C. 
        
        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_List : List[ndarray]
            Dirichlet values to be assigned
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i in range(len(location_fns)):
                node_inds = onp.argwhere(jax.vmap(location_fns[i])(self.mesh.points)).reshape(-1)
                vec_inds = onp.ones_like(node_inds, dtype=onp.int32)*vecs[i]
                values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.
        
        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

    def Neumann_boundary_conditions(self, neumann_bc_info):
        """Indices and values for Neumann B.C. 
        
        Parameters
        ----------
        Neumann_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        """
        node_inds_list = []
        vals_list = []
        if neumann_bc_info is not None:
            location_fns, value_fns = neumann_bc_info
            assert len(location_fns) == len(value_fns)
            for i in range(len(location_fns)):
                node_inds = onp.argwhere(jax.vmap(location_fns[i])(self.mesh.points)).reshape(-1)
                values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
                node_inds_list.append(node_inds)
                vals_list.append(values)
        return [node_inds_list, vals_list]

    def update_Neumann_boundary_conditions(self, neumann_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.
        
        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.neumann_bc_info = self.Neumann_boundary_conditions(neumann_bc_info)



    def periodic_boundary_conditions(self):
        p_node_inds_list_A = []
        p_node_inds_list_B = []
        p_vec_inds_list = []
        if self.periodic_bc_info is not None:
            location_fns_A, location_fns_B, mappings, vecs = self.periodic_bc_info
            for i in range(len(location_fns_A)):
                node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(self.mesh.points)).reshape(-1)
                node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(self.mesh.points)).reshape(-1)
                points_set_A = self.mesh.points[node_inds_A]
                points_set_B = self.mesh.points[node_inds_B]

                EPS = 1e-5
                node_inds_B_ordered = []
                for node_ind in node_inds_A:
                    point_A = self.mesh.points[node_ind]
                    dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
                    node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
                    node_inds_B_ordered.append(node_ind_B_ordered)

                node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
                vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32)*vecs[i]

                p_node_inds_list_A.append(node_inds_A)
                p_node_inds_list_B.append(node_inds_B_ordered)
                p_vec_inds_list.append(vec_inds)
                assert len(node_inds_A) == len(node_inds_B_ordered)

        return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list

    def get_boundary_conditions_inds(self, location_fns):
        """Given location functions, compute which faces satisfy the condition. 
        
        Parameters
        ----------
        location_fns : List[Callable]
            Callable: a function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)
        
        Returns
        -------
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, j] returns the index of face j of cell i of surface k
        """
        cell_points = onp.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_nodes, dim)
        boundary_inds_list = []
        for i in range(len(location_fns)):
            vmap_location_fn = jax.vmap(location_fns[i])
            def on_boundary(cell_points):
                boundary_flag = vmap_location_fn(cell_points)
                return onp.all(boundary_flag)
            vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
            boundary_flags = vvmap_on_boundary(cell_face_points)
            boundary_inds = onp.argwhere(boundary_flags) # (num_selected_faces, 2)
            boundary_inds_list.append(boundary_inds)
        return boundary_inds_list

    def compute_Neumann_integral_vars(self, **internal_vars):
        """In the weak form, we have the Neumann integral: (traction, v) * ds, and this function computes this.

        Returns
        -------
        integral: np.DeviceArray
            (num_total_nodes, vec)
        """
        integral = np.zeros((self.num_total_nodes, self.vec))
        if self.neumann_bc_info is not None:
            for i, boundary_inds in enumerate(self.neumann_boundary_inds_list):  #self.neumann_boundary_inds_list: this is the faceID
                if 'neumann_vars' in internal_vars.keys():
                    int_vars = internal_vars['neumann_vars'][i]
                else:
                    int_vars = ()
                # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
                subset_quad_points = self.get_physical_surface_quad_points(boundary_inds) 
                # int_vars = [x[i] for x in internal_vars]
                traction = jax.vmap(jax.vmap(self.neumann_value_fns[i]))(subset_quad_points, *int_vars) # subset_quad_points dim: (num_selected_faces, num_face_quads, vec)
                assert len(traction.shape) == 3
                _, nanson_scale = self.get_face_shape_grads(boundary_inds) # (num_selected_faces, num_face_quads)
                # (num_faces, num_face_quads, num_nodes): num_faces are the all faces in the parent domain ->  (num_selected_faces, num_face_quads, num_nodes) num_selected_faces: all the faces for all the surface elements
                v_vals = np.take(self.face_shape_vals, boundary_inds[:, 1], axis=0)
                v_vals = np.repeat(v_vals[:, :, :, None], self.vec, axis=-1) # (num_selected_faces, num_face_quads, num_nodes, vec)
                subset_cells = np.take(self.cells, boundary_inds[:, 0], axis=0) # (num_selected_faces, num_nodes)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] * nanson_scale[:, :, None, None], axis=1).reshape(-1, self.vec) #nanson_scale refers to da
                integral = integral.at[subset_cells.reshape(-1)].add(int_vals)   #this part needs to be modified cause C-HiDeNN has interaction within the boundary
        return integral

    def compute_Neumann_integral_vars_CFEM(self, **internal_vars):
        """In the weak form, we have the Neumann integral: (traction, v) * ds, and this function computes this.

        Returns
        -------
        integral: np.DeviceArray
            (num_total_nodes, vec)
        """
        integral = np.zeros((self.num_total_nodes, self.vec)).reshape(-1)
        if self.neumann_bc_info is not None:
            for i, boundary_inds in enumerate(self.neumann_boundary_inds_list):  #self.neumann_boundary_inds_list: this is the faceID
                if 'neumann_vars' in internal_vars.keys():
                    int_vars = internal_vars['neumann_vars'][i]
                else:
                    int_vars = ()
                # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
                subset_quad_points = self.get_physical_surface_quad_points(boundary_inds) 
                # int_vars = [x[i] for x in internal_vars]
                traction = jax.vmap(jax.vmap(self.neumann_value_fns[i]))(subset_quad_points, *int_vars) # subset_quad_points dim: (num_selected_faces, num_face_quads, vec) @ surfaces
                assert len(traction.shape) == 3
                _, nanson_scale = self.get_face_shape_grads(boundary_inds) # (num_selected_faces, num_face_quads)
                # (num_selected_faces, num_face_quads, edex_max)
                v_vals = self.N_til_faces
                v_vals = np.repeat(v_vals[:, :, :, None], self.vec, axis=-1) # (num_selected_faces, num_face_quads, edex_max, vec)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] * nanson_scale[:, :, None, None], axis=1).reshape(-1)
                # print('v_vals', v_vals) 
                # print('traction', traction) 
                # print('Nanson scale', nanson_scale) 
                integral = integral.at[self.face_connectivity_blocks.reshape(-1)].add(int_vals)   #this part needs to be modified cause C-HiDeNN has interaction within the boundary
        return integral

    def compute_Neumann_boundary_inds(self):
        """Child class should override if internal variables exist
        """
        if self.neumann_bc_info is not None:
            self.neumann_location_fns, self.neumann_value_fns = self.neumann_bc_info
            if self.neumann_location_fns is not None:
                self.neumann_boundary_inds_list = self.get_boundary_conditions_inds(self.neumann_location_fns)

    def compute_body_force_by_fn(self):
        """In the weak form, we have (body_force, v) * dx, and this function computes this

        Returns
        -------
        body_force: np.DeviceArray
            (num_total_nodes, vec)
        """
        rhs = np.zeros((self.num_total_nodes, self.vec))
        if self.source_info is not None:
            body_force_fn = self.source_info
            physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
            body_force = jax.vmap(jax.vmap(body_force_fn))(physical_quad_points) # (num_cells, num_quads, vec) 
            assert len(body_force.shape) == 3
            v_vals = np.repeat(self.shape_vals[None, :, :, None], self.num_cells, axis=0) # (num_cells, num_quads, num_nodes, 1)
            v_vals = np.repeat(v_vals, self.vec, axis=-1) # (num_cells, num_quads, num_nodes, vec)
            # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
            rhs_vals = np.sum(v_vals * body_force[:, :, None, :] * self.JxW[:, :, None, None], axis=1).reshape(-1, self.vec) 
            rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals) 
        return rhs

    def compute_body_force_by_sol(self, sol, mass_map):
        """In the weak form, we have (old_solution, v) * dx, and this function computes this(meaning?)
        
        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        mass_map : Callable
            Transformation on sol
        
        Returns
        -------
        body_force : np.DeviceArray
            (num_total_nodes, vec)
        """
        mass_kernel = self.get_mass_kernel(mass_map)
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec) what does this vec mean here?
        val = jax.vmap(mass_kernel)(cells_sol, self.JxW) # (num_cells, num_nodes, vec)
        val = val.reshape(-1, self.vec) # (num_cells*num_nodes, vec)
        body_force = np.zeros_like(sol)
        body_force = body_force.at[self.cells.reshape(-1)].add(val) 
        return body_force 

    ################################################
    #to be changed in material calibration
    def get_laplace_kernel(self, tensor_map): #to be modified in the CFEM code
        def laplace_kernel(cell_sol, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars): #*cell_internal_vars is the parameters
            # cell_sol: (edex, vec)
            # (1, edex, vec, 1) * (num_quads, edex, 1, dim) -> (num_quads, edex, vec, dim) #vec refers to the dimension of the variable, for heat vec = 1, for disp, vec = 3
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :] #parent domain or physical domain cell_sol: nodal u values; cell_shape_grads: \partial N \parial X
            u_grads = np.sum(u_grads, axis=1) # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim) # (num_quads, vec, dim) 
            # (num_quads, vec, dim) 
            ###############################################
            #to be changed in material calibration
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)  #stress \sigma_ij in solid mechanics  dim: (num_quads, vec, dim) 
            # (num_quads, 1, vec, dim) * (num_quads, edex, 1, dim) -> (num_quads, edex, vec, dim) -> sum on 0, -1 ->(num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1)) #internal force fiI [num_edex, vec]
            return val #[num_edex, vec]
        return laplace_kernel  #[num_edex, vec]

    def get_mass_kernel(self, mass_map):
        def mass_kernel(cell_sol, cell_JxW, *cell_internal_vars):
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, *cell_internal_vars) # (num_quads, vec) 
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            return val
        return mass_kernel    

    def get_cauchy_kernel(self, cauchy_map):
        def cauchy_kernel(cell_sol, face_shape_vals, face_nanson_scale):
            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec) 
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(cauchy_map)(u) # (num_face_quads, vec) 
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)
            return val
        return cauchy_kernel

    def unpack_kernels_vars(self, **internal_vars):
        if 'mass' in internal_vars.keys():
            mass_internal_vars = internal_vars['mass']
        else:
            mass_internal_vars = ()

        if 'laplace' in internal_vars.keys():
            laplace_internal_vars = internal_vars['laplace']
        else:
            laplace_internal_vars = ()

        return [mass_internal_vars, laplace_internal_vars]        

    def pre_jit_fns(self):
        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x, ))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(cell_sol_flat, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_mass_internal_vars, cell_laplace_internal_vars):

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_laplace_internal_vars)
                else:
                    laplace_val = 0.

                return laplace_val


            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args) # cell_sol: nodal uiI values; \partial f_iI \partial ujI: tangent stiffness matrix
                return value_and_jacfwd(kernel_partial, cell_sol_flat) # kernel(cell_sol, *args), jax.jacfwd(kernel)(cell_sol, *args) #what derivative is it solving? Answer: tangent stiffness matrix; return fiI and jacobian of fiI

            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
                if hasattr(self, 'get_surface_maps'):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[ind])
                    surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    surface_val = 0.

                return surface_val

            def kernel_jac(cell_sol_flat, *args):
                # return jax.jacfwd(kernel)(cell_sol_flat, *args)
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        self.kernel = kernel
        self.kernel_jac = kernel_jac

        # num_surfaces = len(self.boundary_inds_list)
        # if hasattr(self, 'get_surface_maps'):
        #     assert num_surfaces == len(self.get_surface_maps())
        # elif hasattr(self, 'get_universal_kernels_surface'):
        #     assert num_surfaces == len(self.get_universal_kernels_surface()) 
        # else:
        #     assert num_surfaces == 0, "Missing definitions for surface integral"
            

        # self.kernel_face = []
        # self.kernel_jac_face = []
        # for i in range(len(self.boundary_inds_list)):
        #     kernel_face, kernel_jac_face = get_kernel_fn_face(i)
        #     kernel_face = jax.jit(jax.vmap(kernel_face))
        #     kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
        #     self.kernel_face.append(kernel_face)
        #     self.kernel_jac_face.append(kernel_jac_face)
    
    
    def split_and_compute_cell(self, sol, np_version, jac_flag, **internal_vars): #core of automatic differentiation; 
        '''
        returns: internal force and tangent stiffess for all all all the elements
        values: internal force fiI for different elements: [num_ele, num_nodes, vec]
        jacs: tangent stiffness \partial f_iI \partial ujJ for different elements: [num_ele, num_nodes, vec, num_nodes, vec]
        ### to be modified: sparse matrix return for jacs
        '''
        vmap_fn = self.kernel_jac if jac_flag else self.kernel
        
        def jacs_csr_fun(jacs, connectivity_block): #should be better when I load it by batch
            V_block = jacs.reshape(-1) 
            I_block = onp.repeat(connectivity_block, self.edex_max * self.dim, axis=1).reshape(-1)
            J_block = onp.repeat(connectivity_block, self.edex_max * self.dim, axis=0).reshape(-1)
            #csc_matrix
            jac_csr = scipy.sparse.csr_array((V_block, (I_block, J_block)), shape=(self.num_total_dofs, self.num_total_dofs))
            #print('CSR shape is', jac_csr.nnz)
            # jacs_BCOO = BCOO((V_block, indices_block), shape=(self.num_total_dofs, self.num_total_dofs))
            # jacs_BCOO = jacs_BCOO.sum_duplicates(nse = self.num_total_dofs * self.num_total_dofs)
            
            return jac_csr

        kernal_vars = self.unpack_kernels_vars(**internal_vars) #this line might be related to history-dependent materials
        cells_sol = sol[self.Elemental_patch_nodes_st_blocks]; 
        

        num_cuts = 100 #batch input (how to compute this value for different size problems)
        if num_cuts > len(self.cells): #len(self.cells): num of elements
            num_cuts = len(self.cells)
        batch_size = len(self.cells) // num_cuts


        input_collection = [cells_sol, self.Grad_N_til_blocks, self.JxW_blocks, self.v_grads_JxW_blocks, *kernal_vars]
        if jac_flag:
            values = []
            jacs = []
            cpu = jax.devices("cpu")[0]
            gpu = jax.devices("gpu")[0]
            #mem_report_0(self.gpu_idx)
            for i in range(num_cuts): #loop over each element
                start = time.time()
                if i < num_cuts - 1:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
                else:
                    input_col = jax.tree_map(lambda x: x[i * batch_size:], input_collection)

                val, jac = vmap_fn(*input_col)
                #
                val = jax.device_put(val, gpu)
                jac = jax.device_put(jac, gpu) 
                values.append(val) #append uiI (num_ele, num_nodes, vec)
                jacs.append(jac) #(num_ele, num_nodes, vec, num_nodes, vec) append tangent stiffness \partial f_iI \partial ujJ
                # if i % 25 == 0 :
                #     mem_report(i, self.gpu_idx)
            
            values = np.vstack(jax.device_put(values, gpu)) #stack of fiI for different elements [num_ele, num_nodes, vec]
            jacs = np.vstack(jax.device_put(jacs, gpu)) #stack of tangent stiffness \partial f_iI \partial ujJ for different elements [num_ele, num_nodes, vec, num_nodes, vec]

            #byte_jacs = self.num_total_dofs * self.num_total_dofs * 4
            num_cuts_jac = 1 #int(byte_jacs // 2e9 + 1) 
            if num_cuts_jac  > len(self.cells): #len(self.cells): num of elements
                num_cuts_jac  = len(self.cells)
            batch_size_jac = len(self.cells) // num_cuts_jac
            input_collection_jac = [jacs, self.connectivity_blocks]

            #print(f"Creating sparse matrix directly using JAX BCOO..., num_cuts_jac is", num_cuts_jac)
            start = time.time()
            for i in range(num_cuts_jac): #loop over each jac loading block
                
                if i < num_cuts_jac - 1:
                    input_col_jac = jax.tree_map(lambda x: x[i * batch_size_jac:(i + 1) * batch_size_jac], input_collection_jac)
                else:
                    input_col_jac = jax.tree_map(lambda x: x[i * batch_size_jac:], input_collection_jac)

                jac_csr = jacs_csr_fun(*input_col_jac)
                if i == 0:
                    jacs_csr = jac_csr
                    # print('eye1')
                else:
                    jacs_csr = jacs_csr + jac_csr
            self.A_sp_scipy = jacs_csr
            
            jacs_BCOO = BCOO.from_scipy_sparse(jacs_csr).sort_indices()
            end = time.time()
            #print('Jacobian assembly finished, cost', end - start)
            return values, jacs_BCOO #what's values and jocobians? return of this split and compute function value (8,8,3) local stiffness matrix? jacs 4608
        else:
            values = []
            for i in range(num_cuts): #loop over each element
                if i < num_cuts - 1:
                    input_col = jax.tree_map(lambda x: x[i*batch_size:(i + 1)*batch_size], input_collection)
                else:
                    input_col = jax.tree_map(lambda x: x[i*batch_size:], input_collection)

                val = vmap_fn(*input_col)
                values.append(val) #append uiI (num_ele, num_nodes, vec)
            # np_version set to jax.numpy allows for auto diff, but uses GPU memory
            # np_version set to ordinary numpy saves GPU memory, but can't use auto diff 
            values = np_version.vstack(values) #stack of fiI for different elements [num_ele, num_nodes, vec]
            vmap_fn._clear_cache()
            jax.clear_caches()
            return values #return of this split and compute function



    # def compute_face(self, cells_sol, np_version, jac_flag):
    #     def get_kernel_fn_face(cauchy_map): 
    #         def kernel(cell_sol, face_shape_vals, face_nanson_scale):
    #             cauchy_kernel = self.get_cauchy_kernel(cauchy_map)
    #             val = cauchy_kernel(cell_sol, face_shape_vals, face_nanson_scale)
    #             return val

    #         def kernel_jac(cell_sol, *args):
    #             return jax.jacfwd(kernel)(cell_sol, *args)

    #         return kernel, kernel_jac

    #     # TODO: Better to move the following to __init__ function?
    #     location_fns, value_fns = self.cauchy_bc_info
    #     boundary_inds_list = self.get_boundary_conditions_inds(location_fns)
    #     values = []
    #     selected_cells = []
    #     for i, boundary_inds in enumerate(boundary_inds_list):
    #         selected_cell_sols = cells_sol[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
    #         selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
    #         _, nanson_scale = self.get_face_shape_grads(boundary_inds) # (num_selected_faces, num_face_quads)
    #         kernel, kernel_jac = get_kernel_fn_face(value_fns[i])
    #         fn = kernel_jac if jac_flag else kernel
    #         #vmap_fn = jax.vmap(fn)
    #         vmap_fn = jax.jit(jax.vmap(fn))
    #         val = vmap_fn(selected_cell_sols, selected_face_shape_vals, nanson_scale)   
    #         values.append(val)
    #         selected_cells.append(self.cells[boundary_inds[:, 0]])
        
    #     values = np_version.vstack(values)
    #     selected_cells = onp.vstack(selected_cells)

    #     assert len(values) == len(selected_cells)

    #     return values, selected_cells

    def convert_from_dof_to_quad(self, sol):
        """Obtain quad values from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u : np.DeviceArray
            (num_cells, num_quads, vec)
        """
        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_sol = sol[self.cells] 
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        return u

    def convert_neumann_from_dof(self, sol, index):
        """Obtain surface solution from nodal solution
        
        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        index : int

        Returns
        -------
        u : np.DeviceArray
            (num_selected_faces, num_face_quads, vec) 
        """
        cells_old_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        boundary_inds = self.neumann_boundary_inds_list[index]
        selected_cell_sols = cells_old_sol[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec) 
        u = np.sum(selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        return u

    def compute_residual_vars_helper(self, sol, weak_form, **internal_vars):
        res = np.zeros((self.num_total_nodes, self.vec))
        weak_form = weak_form.reshape(-1, self.vec) # (num_cells * edex, vec) value internal force term
        res = res.at[self.Elemental_patch_nodes_st_blocks.reshape(-1)].add(weak_form) #boundary terms #this part is important, I guess this is trying to assembly the weak form matrix (#internal force matrix) (num_total_nodes*vec) here vec refers to dimension of the problem
        #self.cells.reshape(-1) 1D nodal positions

        if 'body_vars' in internal_vars.keys():
            self.body_force = self.compute_body_force_by_sol(internal_vars['body_vars'], self.get_body_map())

        self.neumann = self.compute_Neumann_integral_vars_CFEM(**internal_vars).reshape(self.num_total_nodes, self.vec)    

        res = res - self.body_force - self.neumann
        self.force_avg = np.average(np.abs(res) + np.abs(self.body_force) + np.abs(self.neumann))
        return res


    def compute_residual_vars(self, sol, **internal_vars): #return the residual force:values
        #print(f"Compute cell residual...")
        #cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        weak_form = self.split_and_compute_cell(sol, np, False, **internal_vars) # (num_cells, num_nodes, vec)  #def split_and_compute_cell(self, cells_sol, np_version, jac_flag, **internal_vars); weak form:fiI(cell_sol, internal_variable)
        return self.compute_residual_vars_helper(sol, weak_form, **internal_vars)
        print('Eye2')     


    def compute_newton_vars(self, sol, **internal_vars):
        #print(f"Compute cell Jacobian and cell residual...")
        cells_sol = sol[self.cells] # I guess this is uiI for all the elements(num_cells, num_nodes, vec)
        ##weak form: internal force fiI; cells_jac: tangent stiffness \partial f_iI \partial ujJ for different elements [num_ele, num_nodes, vec, num_nodes, vec]
        # (num_cells, edex, vec), (num_cells, edex, vec, edex, vec)
        weak_form, A_sp = self.split_and_compute_cell(sol, np, True, **internal_vars)
        
        # print('Eye1') 

        return self.compute_residual_vars_helper(sol, weak_form, **internal_vars), A_sp  #returns the residual force

    def compute_residual(self, sol):  #what's the benefit of this kind of definition?
        return self.compute_residual_vars(sol, **self.internal_vars)

    def newton_update(self, sol): #called by solver_row_elimination
        return self.compute_newton_vars(sol, **self.internal_vars)

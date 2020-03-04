from fenics import *
from mshr import *
import numpy as np
import os
import time
import datetime
import pickle
from subprocess import Popen, PIPE, check_output
from numpy import array, zeros, ones, any, arange, isnan
import ctypes, ctypes.util, numpy, scipy.sparse, scipy.sparse.linalg, collections

parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
 
# Initialize processor ID for parallelized computation and set target folder
processID = MPI.rank(mpi_comm_world())
# get file name
fileName = os.path.splitext(__file__)[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%% Define time constants, reference pressure and geometrical data %%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = 0.0			# Initialize time
t_end = 10.		# End time of the simulation, in s
t_steady = 2.		# Time at which the inlet speed of the fluid gets steady, in s
dt = 5e-3		# Time step of the simulation, in s

pref = 0.0 		# Reference pressure of fluid, in kPa

# Define geometry scales

Length = 500
resolution_dam = 0.25
resolution_water = 0.05
resolution_plane = 0.01

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%% Create mesh from gmsh execution %%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Creating mesh...')

geo_file_content00 = """
Point(1) = {0, 0, 0, 1.0};
Point(2) = {70.2, 0, 0, 1.0};
Point(3) = {22.005, 66.5, 0, 1.0};
Point(4) = {16.425, 103.0, 0, 1.0};
Point(5) = {1.625, 103.0, 0, 1.0};
Point(6) = {1.625, 39.0, 0, 1.0};
"""
chan1_string = "Point(7) = {-%s, 103.0, 0, 1.0};\nPoint(8) = {-%s, 103.0, 0, 1.0};"%(Length,Length+103)
chan2_string = "Point(9) = {-%s, 0, 0, 1.0};"%(Length)
geo_file_content0 = """
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {5, 7};
Line(8) = {7, 8};
Line(9) = {8, 9};
Line(10) = {9, 1};
Line Loop(1) = {-6, -5, 7, 8, 9, 10};          
Line Loop(2) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Surface("Fluid", 6) = {1};
Physical Surface("Solid", 7) = {2}; 
"""
lchar_string = "Characteristic Length {1, 2, 3, 4, 5, 6} = %s;\n"%(1./resolution_dam)
ochar_string = "Characteristic Length {7, 9} = %s;\n"%(1./resolution_water)
pchar_string = "Characteristic Length {8} = %s;\n"%(1./resolution_plane)

geo_file_content = "SetFactory(\"OpenCASCADE\");\n"+geo_file_content00\
                   +chan1_string+chan2_string+geo_file_content0\
                   +lchar_string+ochar_string+pchar_string
                                                  

fname=fileName
with open(fname+".geo", 'w') as f:
    f.write(geo_file_content)
 
cmd = 'gmsh -2 {}.geo -format msh2'.format(fname)
print check_output([cmd], shell=True)  # run in shell mode in case you are not run in terminal
 
cmd = 'dolfin-convert -i gmsh {}.msh {}.xml'.format(fname, fname)
Popen([cmd], stdout=PIPE, shell=True).communicate()
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%% Import mesh and create the submeshes %%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if processID == 0: print('Loading meshes...')
mesh = Mesh(fname+".xml")
if os.path.exists( fname+"_physical_region.xml"):
    subdomains = MeshFunction("size_t", mesh, fname+"_physical_region.xml")
    plot(subdomains)
if os.path.exists( fname+"_facet_region.xml"):
    boundaries = MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    plot(boundaries)
 
# Create 3 submeshes: 
mesh_f_ref = SubMesh(mesh, subdomains, 6)	# Fluid domain in reference configuration
mesh_f_act = SubMesh(mesh, subdomains, 6)	# Fluid domain in actual configuration
mesh_s = SubMesh(mesh, subdomains, 7)		# Structure domain in reference configuration

# Define facets and cells for reference configurations 
# (for the actual configuration, it will be defined in the time loop)
facets_f_ref = MeshFunction('size_t', mesh_f_ref, 1)
facets_s = MeshFunction('size_t', mesh_s, 1)

cells_f_ref = MeshFunction('size_t', mesh_f_ref, 2)
cells_s = MeshFunction('size_t', mesh_s, 2)

# Measuring elements and getting facet normal
da_ref = Measure('ds', domain=mesh_f_ref, subdomain_data=facets_f_ref, metadata={'quadrature_degree': 2})
dv_ref = Measure('dx', domain=mesh_f_ref, subdomain_data=cells_f_ref, metadata={'quadrature_degree': 2})
n_ref = FacetNormal(mesh_f_ref)

dA = Measure('ds', domain=mesh_s, subdomain_data=facets_s, metadata={'quadrature_degree': 2})
dV = Measure('dx', domain=mesh_s, subdomain_data=cells_s, metadata={'quadrature_degree': 2})
N = FacetNormal(mesh_s)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%% Parametrizing boundaries and inflow profile of the fluid %%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Parametrizing boundaries...')


# Fluid domain boundaries
walls    = CompiledSubDomain('near(x[0], -Length) || near(x[1], 0) || near(x[1],Heigth)',Heigth=103.0, Length = Length)
damwall	 = CompiledSubDomain('on_boundary && x[0] >= 0')

# Numerate fluid boundaries
facets_f_ref.set_all(0)
walls.mark(facets_f_ref,3)
damwall.mark(facets_f_ref,5)

File_facets = File("%s.results/Facets_Fluid.pvd"% (fileName))
File_facets << facets_f_ref

# Structure domain boundaries
damdir 	= CompiledSubDomain('near(x[1],0)')
damflu 	= CompiledSubDomain('on_boundary && x[0] <= 1.625')

# Numerate structure boundaries
facets_s.set_all(0)
damflu.mark(facets_s,5)
damdir.mark(facets_s,1)

File_facets = File("%s.results/Facets_Solid.pvd"% (fileName))
File_facets << facets_s


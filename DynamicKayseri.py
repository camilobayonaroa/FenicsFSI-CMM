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

Length = 0.3		# channel length, in m
Heigth = 0.15 		# channel heigth, in m
Level = 0.1 		# water level, in m
xc = 0.15		# x coordinates of the column 
w_col = 0.003		# column width, in m
l_col = 0.08		# column length, in m

resolution_chan = 100
resolution_col = 100

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Physical parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

forces = Constant(('0.0','-10.0'))

rho1=1		# Density, in Kg/m^3
rho2=1e3	# Density, in Kg/m^3
mu1=1e-5        # dynamic viscosity, in kPa s 
mu2=1e-3        # dynamic viscosity, in kPa s 
la = 1.0	        

eps=1e-4
sigma = 720

#theta = [3/2,-2,1/2] 
theta = [1,-1,0] 
#Stabilization constants    

C1 = 2
C2 = 4
C3 = 1
C4 = 0.5

rho_s = 10000     # Density, in Kg/m^3
nu_s = 0.4      # Poisson ration, in 1
mu_s = 2e6    	# in Kg/(ms^2)
lam_s = 2*nu_s*mu_s / (1-2*nu_s)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%% Create mesh from gmsh execution %%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Creating mesh...')

chan_string = "Rectangle(1) = {0, 0, 0, %s, %s, 0};\n"%(Length, Heigth)
col_string = "Rectangle(2) = {%s, %s, 0, %s, %s, 0};" %(xc-0.5*w_col,0.0,w_col,l_col)
geo_file_content0 = """
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
Line(14) = {5, 6};
Curve Loop(1) = {6, 7, 8, 14};
Plane Surface(2) = {1};
Physical Surface("Fluid", 6) = {1};
Physical Surface("Solid", 7) = {2};
"""
lchar_string = "Characteristic Length {5, 6, 7, 8} = %s;\n"%(1./resolution_col)
ochar_string = "Characteristic Length {12, 11, 10, 9} = %s;\n"%(1./resolution_chan)

geo_file_content = "SetFactory(\"OpenCASCADE\");\n"+chan_string+col_string\
                   +geo_file_content0+lchar_string+ochar_string
                                                  

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
walls    = CompiledSubDomain('near(x[0], 0) || near(x[0], Length)|| near(x[1], 0) || near(x[1],Heigth)',Heigth=Heigth, Length = Length)
column 	 = CompiledSubDomain('on_boundary && x[0] >= xo && x[0] <= xf && x[1] >= yo && x[1] <= yf',xo=xc-w_col, xf=xc+w_col,yo=0,yf=l_col)

# Numerate fluid boundaries
facets_f_ref.set_all(0)
walls.mark(facets_f_ref,3)
column.mark(facets_f_ref,5)

File_facets = File("%s.results/Facets_Fluid.pvd"% (fileName))
File_facets << facets_f_ref

# Structure domain boundaries
columndir 	= CompiledSubDomain('near(x[1],0)')
columnflu 	= CompiledSubDomain('on_boundary')

# Numerate structure boundaries
facets_s.set_all(0)
columnflu.mark(facets_s,5)
columndir.mark(facets_s,1)

File_facets = File("%s.results/Facets_Solid.pvd"% (fileName))
File_facets << facets_s


class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = x[1]-Level 
        values[1] = 0.0
        values[2] = 0.0
        values[3] = 0.0
    def value_shape(self):
        return (4,)

ic=InitialCondition(degree =2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Defining some usefull tools %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# indices for tensor calculus, Kroenecker delta in 2D and gravity force (neglegated here)
i, j, k, l, m = indices(5)
delta = Identity(2)

# Lists to save for postprocessing of the data
time_list = [0.]
Xdisplacement_list = [0.]
Ydisplacement_list = [0.]
Lift_list = [0.]
Drag_list = [0.]

# .pvd-files to display results in ParaView
File_u_s = File("%s.results/u.pvd" % (fileName))
File_p_f = File("%s.results/p.pvd" % (fileName))
File_l_f = File("%s.results/l.pvd" % (fileName))
File_v_f = File("%s.results/v.pvd" % (fileName))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%% Defining function spaces for the FEM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if processID == 0: print('Defining function spaces...')

scalar = FiniteElement('P', triangle, 1)
vector = VectorElement('P', triangle, 1)
tensor = TensorElement('P', triangle, 1)
mixed_element = MixedElement([scalar, scalar, vector])

MixedSpace_f = FunctionSpace(mesh_f_act, mixed_element)

E_f_ref = FunctionSpace(mesh_f_ref, scalar)
V_f_ref = FunctionSpace(mesh_f_ref, vector)
T_f_ref = FunctionSpace(mesh_f_ref, tensor)

S_s_space = FunctionSpace(mesh_s, scalar)
V_s_space = FunctionSpace(mesh_s, vector)
T_s_space = FunctionSpace(mesh_s, tensor)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%% Defining functions, test functions and trial functions %%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_s = Function(V_s_space)		# Structure displacement at t
uk_s = Function(V_s_space)		# Structure displacement at t, previous for-iteration
u0_s = Function(V_s_space)		# Structure displacement at t-dt
u00_s = Function(V_s_space)		# Structure displacement at t-2*dt	
del_u = TestFunction(V_s_space)		# Test function for structure displacement
du = TrialFunction(V_s_space)		# Trial function for structure displacement

u_m = Function(V_f_ref)			# Mesh displacement at t
uk_m = Function(V_f_ref)		# Mesh displacement at t, previous for-iteration
u0_m = Function(V_f_ref)		# Mesh displacement at t-dt
del_u_m = TestFunction(V_f_ref)		# Test function for mesh displacement
du_m = TrialFunction(V_f_ref)		# Trial function for mesh displacement

u_f = Function(MixedSpace_f)		# Fluid levelset, pressure and velocity at t 
u0_f = Function(MixedSpace_f)		# Fluid levelset, pressure and velocity at t-dt
u00_f = Function(MixedSpace_f)		# Fluid levelset, pressure and velocity at t-2*dt
del_u_f = TestFunction(MixedSpace_f)	# Test function for fluid levelset, pressure and velocity
du_f = TrialFunction(MixedSpace_f)	# Trial function for fluid levelset, pressure and velocity

# Split the mixed function for fluid FEM
l, p, v = split(u_f)
l0, p0, v0 = split(u0_f)
l00, p00, v00 = split(u00_f)
del_l, del_p, del_v = split(del_u_f)
dl_f, dp_f, dv_f = split(du_f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize solutions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_f.interpolate(ic)
u0_f.assign(u_f)
u00_f.assign(u_f)

u_s_init = Expression( ('0.0','0.0') , degree=0)
u_s.interpolate(u_s_init)
u0_s.assign(u_s)
u00_s.assign(u_s)

t_s_hat = Constant((0.0, 0.0))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Constants, tensors, and boundary conditions for structure %%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F_s = as_tensor( u_s[k].dx(i) + delta[k,i], (k,i) )
J_s = det(F_s)
C_s = as_tensor( F_s[k,i]*F_s[k,j], (i,j) )
E_s = as_tensor( 1./2.*(C_s[i,j]-delta[i,j]), (i,j) )
S_s = as_tensor( lam_s*E_s[k,k]*delta[i,j] + 2.*mu_s*E_s[i,j], (i,j) )
P_s = as_tensor( F_s[i,j]*S_s[j,k], (k,i) )

bc_s = []

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Constants, tensors, and boundary conditions for structure %%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nu_m = 0.0    				# Poisson ratio, in 1
mu_m = 1000/CellDiameter(mesh_f_ref)**4	# Stifness inverse proportional to cell size, in in ton/(ms^2)
la_m = 2*nu_m*mu_m / (1-2*nu_m)

eps_m = as_tensor(1.0/2.0*(u_m[i].dx(j)+u_m[j].dx(i)), (i,j))
sigma_m = as_tensor( la_m*eps_m[k,k]*delta[i,j] + 2.0*mu_m*eps_m[i,j], (i,j))
a_m = sigma_m[j,i]*del_u_m[i].dx(j)*dv_ref
L_m = 0.0 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fluid stabilization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def advnorm(tempmesh,uk,cvel):
        unorm = project(sqrt((uk-cvel)**2),FunctionSpace(tempmesh, 'P', 1))
        chale = sqrt(dot(dot(uk-cvel,mesh_metric(tempmesh)),uk-cvel)/(dot(uk-cvel,uk-cvel)+DOLFIN_EPS))
        invchale = conditional(lt(chale,1E-4),1.0E12,1.0/chale)
        return unorm*invchale

def sym2asym(HH):
    if HH.shape[0] == 3:
        return array([HH[0,:],HH[1,:],\
                      HH[1,:],HH[2,:]])
    else:
        return array([HH[0,:],HH[1,:],HH[3,:],\
                      HH[1,:],HH[2,:],HH[4,:],\
                      HH[3,:],HH[4,:],HH[5,:]])

def c_cell_dofs(mesh,V):
   if V.ufl_element().is_cellwise_constant():
    return arange(mesh.num_cells()*mesh.geometry().dim()**2)
   else:
    return arange(mesh.num_vertices()*mesh.geometry().dim()**2)

def mesh_metric(mesh):
        # this function calculates a mesh metric (or perhaps a square inverse of that, see mesh_metric2...)
        cell2dof = c_cell_dofs(mesh,TensorFunctionSpace(mesh, "DG", 0))
        cells = mesh.cells()
        coords = mesh.coordinates()
        p1 = coords[cells[:,0],:]
        p2 = coords[cells[:,1],:]
        p3 = coords[cells[:,2],:]
        r1 = p1-p2; r2 = p1-p3; r3 = p2-p3
        Nedg = 3
        if mesh.geometry().dim() == 3:
          Nedg = 6
          p4 = coords[cells[:,3],:]
          r4 = p1-p4; r5 = p2-p4; r6 = p3-p4
        rall = zeros([p1.shape[0],p1.shape[1],Nedg])
        rall[:,:,0] = r1; rall[:,:,1] = r2; rall[:,:,2] = r3
        if mesh.geometry().dim() == 3:
          rall[:,:,3] = r4; rall[:,:,4] = r5; rall[:,:,5] = r6
        All = zeros([p1.shape[0],Nedg**2])
        inds = arange(Nedg**2).reshape([Nedg,Nedg])
        for i in range(Nedg):
          All[:,inds[i,0]] = rall[:,0,i]**2; All[:,inds[i,1]] = 2.*rall[:,0,i]*rall[:,1,i]; All[:,inds[i,2]] = rall[:,1,i]**2
          if mesh.geometry().dim() == 3:
            All[:,inds[i,3]] = 2.*rall[:,0,i]*rall[:,2,i]; All[:,inds[i,4]] = 2.*rall[:,1,i]*rall[:,2,i]; All[:,inds[i,5]] = rall[:,2,i]**2
        Ain = zeros([Nedg*2-1,Nedg*p1.shape[0]])
        ndia = zeros(Nedg*2-1)
        for i in range(Nedg):
          for j in range(i,Nedg):
              iks1 = arange(j,Ain.shape[1],Nedg)
              if i==0:
                  Ain[i,iks1] = All[:,inds[j,j]]
              else:
                  iks2 = arange(j-i,Ain.shape[1],Nedg)
                  Ain[2*i-1,iks1] = All[:,inds[j-i,j]]
                  Ain[2*i,iks2]   = All[:,inds[j,j-i]]
                  ndia[2*i-1] = i
                  ndia[2*i]   = -i
        
        A = scipy.sparse.spdiags(Ain, ndia, Ain.shape[1], Ain.shape[1]).tocsr()
        b = ones(Ain.shape[1])
        X = scipy.sparse.linalg.spsolve(A,b)
        #set solution
        XX = sym2asym(X.reshape([mesh.num_cells(),Nedg]).transpose())
        M = Function(TensorFunctionSpace(mesh,"DG", 0))
        M.vector().set_local(XX.transpose().flatten()[cell2dof])
        return M

def Sign(q):
    return conditional(lt(abs(q),eps),q/eps,sign(q))

def rho(l):
    return(rho1*0.5*(1.0+Sign(l)) + rho2*0.5*(1.0-Sign(l)))

def mu(l):
   return(mu1*0.5*(1.0+Sign(l)) + mu2*0.5*(1.0-Sign(l)))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Time simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t1 = datetime.datetime.now()	# initialize time counting of simulation 
if processID == 0: print('Starting transient simulation... \n')

while t < t_end:

	# Update time and inflow profile
	tic()
	t += dt 

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%% Beginning of global FOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	L2_abs = 1.0			# initialize L2-norm for structure deformation
	L2_rel = 1.0
	abs_tol = 1e-9
	min_trial = 2
	count = 0

	uk_s.assign(u0_s)
	uk_m.assign(u0_m)
        #Ok, there should be reference meshes in order to perform boundary communications.
   
        #Carefull with dynamic dirichlet boundary conditions
        #bc_s.append( DirichletBC(V_s_space, Expression(('cos(t)', '0'), t=t, degree=3), facets_s, 1) )
        bc_s.append( DirichletBC(V_s_space, Expression(('0', '0'), t=t, degree=3), facets_s, 1) )

	while count < min_trial or L2_abs > abs_tol:
	
		count += 1
		if processID == 0: print '\n', 'Solving step:', count, ', at time: ', t, '\n\n'
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve structure deformation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		if processID == 0: print '     Solving STRUCTURE deformation...', '\n'

		Form_s1 = rho_s*((u_s-2.*u0_s+u00_s)[i]/(dt*dt)*del_u[i])*dV
		Form_s2 = rho_s*(P_s[k,i]*del_u[i].dx(k)-forces[i]*del_u[i])*dV
                Form_s3 = -t_s_hat[i]*del_u[i]*dA
                Form_s = Form_s1 + Form_s2 + Form_s3

		Gain_s = derivative(Form_s, u_s, du)

		solve(Form_s == 0, u_s, bc_s, J=Gain_s, \
			solver_parameters={"newton_solver":{"linear_solver": "mumps","absolute_tolerance": 1e-6, "relative_tolerance": 1e-3, "maximum_iterations":10000}},
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})			

		# Project the solution on reference configuration of fluid domain, in order to give the structure displacement as DirichletBC for mesh displacement
		u_m_bound = project(u_s, V_f_ref, solver_type="mumps", \
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})


		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve fluid's mesh displacement %%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		 
		if processID == 0: print '\n', '     Solving FLUID DOMAIN displacement...', '\n'

		# Define voundary conditions for mesh displacement
		bc_m = []

		#bc_m.append( DirichletBC(V_f_ref, Expression(('cos(t)', '0'), t=t, degree=3), facets_f_ref, 3) )
		bc_m.append( DirichletBC(V_f_ref, Expression(('0', '0'), t=t, degree=3), facets_f_ref, 3) )
		bc_m.append( DirichletBC(V_f_ref, u_m_bound, facets_f_ref, 5) )
		

		# Get the pressure and velocity value at t-dt
		mesh_f_old = Mesh(mesh_f_act)
		MixedSpace_f_old = FunctionSpace(mesh_f_old, mixed_element)
		u0_f_old = Function(MixedSpace_f_old)
		u0_f_old.assign(u0_f)
		u00_f_old = Function(MixedSpace_f_old)
		u00_f_old.assign(u00_f)

		# Get new mesh displacement
		try: solve(a_m==L_m, u_m, bcs=bc_m)
		except: 
			u_m.assign(uk_m)
			if processID == 0: print '     CAUTION: I had to do an exception...'

		# Get mesh velocity	
		v_m_ref = project((u_m-u0_m)/dt, V_f_ref, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"})
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%% Build new mesh for the actual configuration %%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		if processID == 0: print '     Building new mesh...'

		mesh_f_act = Mesh(mesh_f_ref)

		facets_f_act = MeshFunction('size_t', mesh_f_act, 1)
		cells_f_act = MeshFunction('size_t', mesh_f_act, 2)

		da_act = Measure('ds', domain=mesh_f_act, subdomain_data=facets_f_act, metadata={'quadrature_degree': 2})
		dv_act = Measure('dx', domain=mesh_f_act, subdomain_data=cells_f_act, metadata={'quadrature_degree': 2})
		n_act = FacetNormal(mesh_f_act)

		facets_f_act.set_all(0)
                walls.mark(facets_f_act,3)
                column.mark(facets_f_act,5)

		MixedSpace_f = FunctionSpace(mesh_f_act, mixed_element)
		E_f_act = FunctionSpace(mesh_f_act, scalar)
		V_f_act = FunctionSpace(mesh_f_act, vector)
		T_f_act = FunctionSpace(mesh_f_act, tensor)

		# Define functions for fluid on the new mesh
		du_f = TrialFunction(MixedSpace_f)
		del_u_f = TestFunction(MixedSpace_f)
		u_f = Function(MixedSpace_f)
		u0_f = Function(MixedSpace_f)
		u00_f = Function(MixedSpace_f)
		uk_f = Function(MixedSpace_f)

		l, p, v = split(u_f)
		dl_f, dp_f, dv_f = split(du_f)
		del_l, del_p, del_v = split(del_u_f)

		# Deform the new mesh
		for x in mesh_f_act.coordinates(): x[:] += u_m(x)[:]
		mesh_f_act.bounding_box_tree().build(mesh_f_act)

		# Pass the mesh velocity values to the current mesh configuration
		v_m_act = Function(V_f_act)
		v_m_act.vector()[:] = v_m_ref.vector().get_local()

		# Pass the fluid pressure and velocity to the current mesh configuration
		u0_f.assign( project(u0_f_old , MixedSpace_f , solver_type="mumps", \
		    form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}) )
		u00_f.assign( project(u00_f_old , MixedSpace_f , solver_type="mumps", \
		    form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}) )

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%% Solve fluid dynamic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                #Constant Stabilization terms
                hlen = project(CellDiameter(mesh_f_act), E_f_act)

		if processID == 0: print '\n', '     Solving FLUID FLOW dynamics...', '\n'

		#%% Define boundary conditions for the fluid
		bc_f = []
		bc_f.append( DirichletBC(MixedSpace_f.sub(2), Constant((0, 0)), facets_f_act, 3) )
		bc_f.append( DirichletBC(MixedSpace_f.sub(2), Constant((0, 0)), facets_f_act, 5) )
		
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Picard Iteration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		L2_error = 1.0  # error measure ||u-u_k||
		tol = 1e-5     	# tolerance
		it = 0          # iteration counter
		maxiter = 3	# iteration limit		
		
		uk_f.assign(u0_f)
		l0, p0, v0 = split(u0_f)
		l00, p00, v00 = split(u00_f)
		
                #Calculate linear variational terms
                F_aa= theta[1]*rho(l0)*v0[j]/dt*del_v[j]*dv_act 
                F_aaa= theta[2]*rho(l00)*v00[j]/dt*del_v[j]*dv_act 
                F_ll= theta[1]*l0/dt*del_l*dv_act 
                F_lll= theta[2]*l00/dt*del_l*dv_act 

		while L2_error > tol and it < maxiter:
			
			it += 1				
			l_k, p_k, v_k = split(uk_f)
			d = as_tensor( 1./2.*(dv_f[i].dx(j)+dv_f[j].dx(i)) , [i,j] )
			tau = as_tensor(la*d[k,k]*delta[i,j] + 2.*mu(l_k)*d[i,j] , [i,j] )
                        #Calculate variational subscale terms
			Luu = as_tensor(rho(l_k)*((v_k[i]-v_m_act[i])*dv_f[j]).dx(i),[j])
			Lup = as_tensor(dp_f.dx(j),[j])
			Lul = as_tensor(sigma*(l_k.dx(i)/sqrt(l_k.dx(k)*l_k.dx(k))).dx(i)*dl_f.dx(j),[j])
                        Lu  = Luu + Lup + Lul
			Luv = as_tensor(rho(l_k)*((v_k[i]-v_m_act[i])*del_v[j]).dx(i),[j])
			Luq = as_tensor(del_p.dx(j),[j])
			Lun = as_tensor(sigma*(l_k.dx(i)/sqrt(l_k.dx(k)*l_k.dx(k))).dx(i)*del_l.dx(j),[j])
                        Lv  = Luv + Luq + Lun
                        Lqu = dv_f[i].dx(i)
                        Lqv = del_v[i].dx(i)
			Lll = ((v_k[i]-v_m_act[i])*dl_f).dx(i)
			Lln = ((v_k[i]-v_m_act[i])*del_l).dx(i)
                        F_f = as_vector([0, -rho(l)*9.81, 0, 0])
                       
			
                        #Calculate finite scale variational terms
		        Forces = -rho(l_k)*(forces[j]*del_v[j])*dv_act
                        F_a= theta[0]*rho(l_k)*dv_f[j]/dt*del_v[j]*dv_act 
                        F_l= theta[0]*dl_f/dt*del_l*dv_act 
			F_b = dv_f[i].dx(i)*del_p*dv_act
			F_c = rho(l_k)*((v_k[i]-v_m_act[i])*dv_f[j]).dx(i)*del_v[j]*dv_act
			F_d = (dp_f.dx(j)*del_v[j]+tau[i,j]*del_v[j].dx(i))*dv_act
			F_k = sigma*(l_k.dx(i)/sqrt(l_k.dx(k)*l_k.dx(k))).dx(i)*dl_f.dx(j)*del_v[j]*dv_act
			F_n = ((v_k[i]-v_m_act[i])*dl_f).dx(i)*del_l*dv_act

                        #Calculate stabilization term
                        norminvcha = advnorm(mesh_f_act,v_k,v_m_act)
                        freq0 = project(Constant(C1)*mu(l_k)/(hlen*hlen)+Constant(C4)*rho(l_k)/dt,E_f_act)    
                        freq1 = project(Constant(C2)*rho(l_k)*norminvcha,E_f_act)   
                        freto = freq0 + freq1                 
                        freqcl = project(Constant(C2)*norminvcha,E_f_act)
                        freqtl = Constant(C4/dt)              
                        fretol = freqcl + freqtl
                        timom = project(conditional(lt(freto,1E-4),1.0E12,1.0/freto), E_f_act)
                        tidiv = project(Constant(C3)*hlen*hlen/timom, E_f_act)
                        tilev = project(conditional(lt(fretol,1E-4),1.0E12,1.0/fretol), E_f_act)
                        F_stab = (timom*Lu[j]*Lv[j]+tidiv*Lqu*Lqv+tilev*Lll*Lln)*dv_act
                        F_stabf = -timom*rho(l_k)*forces[j]*Lv[j]*dv_act

			Form_f = F_aaa + F_aa + F_a + F_lll + F_ll + F_l + F_b + F_c + F_d + F_k + F_n + F_stab + F_stabf + Forces
			a_f = lhs(Form_f)
			L_f = rhs(Form_f)

			solve(a_f == L_f, u_f, bcs=bc_f, solver_parameters = {"linear_solver":"mumps"}) 
			L2_error = assemble(((u_f-uk_f)**2)*dx)
			if processID == 0: print '          it=%d: L2-error=%g' % (it, L2_error)
			uk_f.assign(u_f)
			
			if it == maxiter and processID == 0: print 'Solver did not converge!'

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Picard Iteration - End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		l_ = u_f.split(deepcopy=True)[0]
		p_ = u_f.split(deepcopy=True)[1]
		v_ = u_f.split(deepcopy=True)[2]

		# Compute the L2-Norm of the structure	
		L2_abs = assemble((((u_s-uk_s)**2.0)**0.5)*dx)
		uk_vec = uk_s.vector()
		us_vec = u_s.vector()
		# if processID == 0: print uk_vec, us_vec
		
		# Compute fluid stress on current configuration
		d_ = as_tensor( 1./2.*( v_[i].dx(j)+v_[j].dx(i) ) , [i,j] )
		tau_ = as_tensor( la*d_[k,k]*delta[i,j] + 2.*mu(l_)*d_[i,j] , [i,j] )
		sigma_f_act = project( -p_*delta+tau_ , T_f_act, solver_type="mumps", \
		 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )

		# Pass the values to the reference configurations
		sigma_f_ref = Function(T_f_ref)
		sigma_f_ref.vector()[:] = sigma_f_act.vector().get_local()

		# Project the values on the structure domain in order to get the traction vector
		sigma_s = project( sigma_f_ref , T_s_space, solver_type="mumps", \
		 	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs"} )

		P_f_ref = as_tensor( J_s * inv(F_s)[k,j] * sigma_s[j,i], (k,i) )
		t_s_hat = as_tensor( N[k] * P_f_ref[k,i], (i,) )
                Hforce = t_s_hat[0]*dA
                Vforce = t_s_hat[1]*dA
                Drag = assemble(Hforce)
                Lift = assemble(Vforce) 

		if processID == 0: 
			print '     Absolute L2-norm for convergence:', L2_abs 
			# print '     Relative L2-norm for convergence:', L2_rel 
			print '     Tip deflection (in m):', u_s(Point(0.6,0.2))

		# Assign values for next FOR-Iteration
		uk_m.assign(u_m)
		uk_s.assign(u_s)


	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%% End of global FOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	if processID == 0: print '\n'

	u00_s.assign(u0_s)
	u0_s.assign(u_s)
	u0_m.assign(u_m)
	u0_f.assign(u_f)

	# Solve results as .pvd 
	#if round(t*1e7)%1000 == 0:
	l_save = u_f.split(deepcopy=True)[0]
	p_save = u_f.split(deepcopy=True)[1]
	v_save = u_f.split(deepcopy=True)[2]

	l_save.rename("l", "tmp")
	p_save.rename("p", "tmp")
	v_save.rename("v", "tmp")
	u_s.rename("u", "tmp")

	File_u_s << (u_s, round(t*1000)/1000.0)
	File_l_f << (l_save, round(t*1000)/1000.0)
	File_p_f << (p_save, round(t*1000)/1000.0)
	File_v_f << (v_save, round(t*1000)/1000.0)
	
	if processID == 0: print 'Results saved'

	time_list.append(t)
	Xdisplacement_list.append( u_s(Point(xc+0.5*w_col,l_col))[0] )
	Ydisplacement_list.append( u_s(Point(xc+0.5*w_col,l_col))[1] )
        Drag_list.append(Drag)
        Lift_list.append(Lift)

        if processID == 0: print ("A_xdisp = %e     A_ydisp = %e      Drag= %e    Lift= %e" % (u_s(Point(xc+0.5*w_col,l_col))[0], u_s(Point(xc+0.5*w_col,l_col))[1], Drag , Lift))

if processID == 0: print 'Simulation took: ', datetime.datetime.now() - t1, 's'

numpy.savetxt("TurekDisplacements.csv", numpy.c_[time_list, Xdisplacement_list, Ydisplacement_list], delimiter=",")
numpy.savetxt("TurekForces.csv", numpy.c_[time_list, Drag_list, Lift_list], delimiter=",")

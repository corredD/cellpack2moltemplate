#!/usr/bin/env python

# Authors: Andrew Jewett (Scripps) and Ludovic Autin (Scripps)
# License: 3-clause BSD License  (See LICENSE.md)
# Copyright (c) 2018, Scripps Research Institute
# All rights reserved.

"""
cellpack2lt.py converts json formatted files created by CellPACK into
moltemplate format.
"""
import sys, json
from collections import defaultdict
import numpy as np
from math import *
if sys.version < '2.7':
    from ordereddict import OrderedDict
else:
    from collections import OrderedDict 
import transformation as tr
g_program_name = __file__.split('/')[-1]   # = 'cellpack2lt.py'
__version__ = '0.2.1'
__date__ = '2018-8-13'

g_control_vmd_colors = False
g_ingredient_id=0

doc_msg = \
    'Typical Usage:\n\n' + \
    '  ' + g_program_name + ' -in HIV-1_0.1.cpr -out system.lt\n' + \
    ' or \n' + \
    '  ' + g_program_name + '  <  HIV-1_0.1.cpr  >   system.lt\n' + \
    '\n' + \
    'where \"HIV-1_0.1.cpr\" is a JSON file in CellPACK output format,\n' + \
    '  and \"HIV-1_0.1.lt\" is the corresponding file converted to moltemplate format\n' + \
    'Optional Arguments\n' + \
    '   -in FILE_NAME     # Specify the name of the input file (as opposed to \"<\")\n' + \
    '   -url URL          # Read the CellPACK JSON text from URL instead of a file\n' +\
    '   -out FILE_NAME    # Specify the output file (as opposed to \">\")\n' + \
    '   -pairstyle STYLE  # Select force formula (eg. -pairstyle lj/cut/coul/debye)\n' + \
    '   -debye            # Specify the Debye length (if applicable)\n' +\
    '   -epsilon          # Specify the \"Epsilon\" Lennard-Jones coeff (default: 1)\n' + \
    '   -deltaR           # Specify the resolution of particle radii (default 0.1)\n' + \
    '   -name OBJECTNAME  # Create a moltemplate object which contains everything.\n' + \
    '                     # (useful if you want multiple copies of the system later)\n'

import sys, json
from collections import defaultdict
import numpy as np
from math import *


class InputError(Exception):
    """ A generic exception object containing a string for error reporting.
        (Raising this exception implies that the caller has provided
         a faulty input file or argument.)

    """

    def __init__(self, err_msg):
        self.err_msg = err_msg

    def __str__(self):
        return self.err_msg

    def __repr__(self):
        return str(self)


def Quaternion2Matrix(q, M):
    "convert a quaternion q to a 3x3 rotation matrix M"""

    M[0][0] =  (q[0]*q[0])-(q[1]*q[1])-(q[2]*q[2])+(q[3]*q[3])
    M[1][1] = -(q[0]*q[0])+(q[1]*q[1])-(q[2]*q[2])+(q[3]*q[3])
    M[2][2] = -(q[0]*q[0])-(q[1]*q[1])+(q[2]*q[2])+(q[3]*q[3])
    M[0][1] = 2*(q[0]*q[1] - q[2]*q[3]);
    M[1][0] = 2*(q[0]*q[1] + q[2]*q[3]);
    M[1][2] = 2*(q[1]*q[2] - q[0]*q[3]);
    M[2][1] = 2*(q[1]*q[2] + q[0]*q[3]);
    M[0][2] = 2*(q[0]*q[2] + q[1]*q[3]);
    M[2][0] = 2*(q[0]*q[2] - q[1]*q[3]);



def AffineTransformQ(x_new, x, q, deltaX, M=None):
    """ This function performs an affine transformation on vector "x".
        Multiply 3-dimensional vector "x" by first three columns of 3x3 rotation
        matrix M.  Add to this the final column of M.  Store result in "dest":
    x_new[0] = M[0][0]*x[0] + M[0][1]*x[1] + M[0][2]*x[2]  +  deltaX[0]
    x_new[1] = M[1][0]*x[0] + M[1][1]*x[1] + M[1][2]*x[2]  +  deltaX[1]
    x_new[2] = M[2][0]*x[0] + M[2][1]*x[1] + M[2][2]*x[2]  +  deltaX[2]
        where the rotation matrix, M, is determined by the quaternion, q.
        Optionally, the user can supply a preallocated 3x3 array (M)

    """
    if M == None:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert((M.shape[0] == 3) and (M.shape[1] == 3))

    Quaternion2Matrix(q, M)

    # There's probably a faster way to do this in numpy without using a for loop
    # but I'm too lazy to lookup what it is
    for i in range(0, 3):
        x_new[i] = 0.0
        for j in range(0, 3):
            x_new[i] += M[i][j] * x[j]
        x_new[i] += deltaX[i]  # (translation offset stored in final column)

def ApplyMatrix(coords,mat):
    """
    Apply the 4x4 transformation matrix to the given list of 3d points.

    @type  coords: array
    @param coords: the list of point to transform.
    @type  mat: 4x4array
    @param mat: the matrix to apply to the 3d points

    @rtype:   array
    @return:  the transformed list of 3d points
    """

    #4x4matrix"
    mat = np.array(mat)
    coords = np.array(coords)
    one = np.ones( (coords.shape[0], 1), coords.dtype.char )
    c = np.concatenate( (coords, one), 1 )
    return np.dot(c, np.transpose(mat))[:, :3]

def AdjustBounds(bounds, X):
    """ 
    Check if a coordinate lies within the box boundary.
    If not, update the box boundary.
    """
    assert(len(bounds) == 2)
    assert(len(bounds[0]) == len(bounds[1]) == len(X))

    for d in range(0, len(X)):
        if bounds[1][d] < bounds[0][d]:
            bounds[0][d] = bounds[1][d] = X[d]
        else:
            if X[d] < bounds[0][d]:
                bounds[0][d] = X[d]
            if bounds[1][d] < X[d]:
                bounds[1][d] = X[d]



def RadiiNeeded(tree,
                ir_needed,
                delta_r):
    """
    Note: LAMMPS (and most other) molecular dynamics simulation programs does
          not allow users to define unique radii for each of the 10^6 or so
          atoms in a simulation.  LAMMPS only allows for about 10^3 different
          -types- of particles in a simulation.  Each of the particle types
          in a CellPack file potentially has a unique radius, so we quantize
          each radius (by dividing it by delta_r and rounding) so that
          particles with similar radii are assigned to the same type.
          The first step is to figure out how many different radii of particles
          will be needed.  Do that by scanning the JSON file.

    This function recursively searches for objects in the JSON file containing
    a 'radii' field.  Quantize each of the radii (by dividing by delta_r) and 
    add them to the set of radii that we will need to define later (ir_needed).
    """

    if not isinstance(tree, dict):
        return
    if 'radii' in tree:
        r_ni = tree['radii']
        for n in range(0, len(tree['radii'])): #loop over "subunits" of this molecule
            for i in range(0, len(r_ni[n]['radii'])):   #loop over atoms
                iradius = int(round(r_ni[n]['radii'][i]/delta_r))  #(quantize the radii)
                ir_needed.add(iradius)
    else:
        for object_name in tree:
            if object_name == "mb" : continue
            RadiiNeeded(tree[object_name], ir_needed, delta_r)


def qnormalize(q):
    norm = 1.0 / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    q[0] *= norm;
    q[1] *= norm;
    q[2] *= norm;
    q[3] *= norm;
    return q

def exyz_to_q(ex, ey, ez):
    # squares of quaternion components
    q0sq = 0.25 * (ex[0] + ey[1] + ez[2] + 1.0);
    q1sq = q0sq - 0.5 * (ey[1] + ez[2]);
    q2sq = q0sq - 0.5 * (ex[0] + ez[2]);
    q3sq = q0sq - 0.5 * (ex[0] + ey[1]);
    # some component must be greater than 1/4 since they sum to 1
    # compute other components from it
    q=[0,0,0,1]
    if (q0sq >= 0.25) :
        q[0] = sqrt(q0sq);
        q[1] = (ey[2] - ez[1]) / (4.0*q[0]);
        q[2] = (ez[0] - ex[2]) / (4.0*q[0]);
        q[3] = (ex[1] - ey[0]) / (4.0*q[0]);
    elif (q1sq >= 0.25) :
        q[1] = sqrt(q1sq);
        q[0] = (ey[2] - ez[1]) / (4.0*q[1]);
        q[2] = (ey[0] + ex[1]) / (4.0*q[1]);
        q[3] = (ex[2] + ez[0]) / (4.0*q[1]);
    elif (q2sq >= 0.25) :
        q[2] = sqrt(q2sq);
        q[0] = (ez[0] - ex[2]) / (4.0*q[2]);
        q[1] = (ey[0] + ex[1]) / (4.0*q[2]);
        q[3] = (ez[1] + ey[2]) / (4.0*q[2]);
    elif (q3sq >= 0.25) :
        q[3] = sqrt(q3sq);
        q[0] = (ex[1] - ey[0]) / (4.0*q[3]);
        q[1] = (ez[0] + ex[2]) / (4.0*q[3]);
        q[2] = (ez[1] + ey[2]) / (4.0*q[3]);
    return qnormalize(q);

MAXJACOBI = 51
def rotate(matrix,  i,  j,  k,  l, s,  tau):#matrix[3][3]
    g = matrix[i][j];
    h = matrix[k][l];
    matrix[i][j] = g-s*(h+g*tau);
    matrix[k][l] = h+s*(g-h*tau);
    return matrix

def jacobi(matrix):
    #double *evalues, double evectors[3][3]
    evalues=np.zeros(3)
    evectors=np.zeros((3,3))
    #int i,j,k;
    #double tresh,theta,tau,t,sm,s,h,g,c,b[3],z[3];
    b=[0,0,0]
    z=[0,0,0]
    tresh = 0.0
    h=0
    g=0
    theta=0
    tau=0
    t=0
    s=0
    c=0
    for i in range(3):
        for j in range(3): 
            evectors[i][j] = 0.0;
        evectors[i][i] = 1.0;

    for i in range(3):
        b[i] =  matrix[i][i];
        evalues[i] = matrix[i][i];
        z[i] = 0.0;

    for iter in range(1,MAXJACOBI):#(int iter = 1; iter <= MAXJACOBI; iter++) {
        sm = 0.0;
        for i in range(2):
            for j in range(i+1,3): 
                 sm += fabs(matrix[i][j]);
        if (sm == 0.0) :
            return [0,evalues,evectors];
        if (iter < 4): tresh = 0.2*sm/(3*3);
        else : tresh = 0.0;
        for i in range(2):
            for j in range(i+1,3): 
                g = 100.0*fabs(matrix[i][j]);
                if (iter > 4 and fabs(evalues[i])+g == fabs(evalues[i]) and fabs(evalues[j])+g == fabs(evalues[j])):
                    matrix[i][j] = 0.0;
                elif (fabs(matrix[i][j]) > tresh) :
                    h = evalues[j]-evalues[i];
                    if (fabs(h)+g == fabs(h)):
                        t = (matrix[i][j])/h;
                    else :
                        theta = 0.5*h/(matrix[i][j]);
                        t = 1.0/(fabs(theta)+sqrt(1.0+theta*theta));
                        if (theta < 0.0): t = -t;
                    c = 1.0/sqrt(1.0+t*t);
                    s = t*c;
                    tau = s/(1.0+c);
                    h = t*matrix[i][j];
                    z[i] -= h;
                    z[j] += h;
                    evalues[i] -= h;
                    evalues[j] += h;
                    matrix[i][j] = 0.0;
                    for k in range(0,i):   rotate(matrix,k,i,k,j,s,tau);
                    for k in range(i+1,j): rotate(matrix,i,k,k,j,s,tau);
                    for k in range(j+1,3): rotate(matrix,i,k,j,k,s,tau);
                    for k in range(0,3):   rotate(evectors,k,i,k,j,s,tau);
    for i in range(3):
        b[i] += z[i]
        evalues[i]+= z[i] ;
        z[i] = 0.0;
    return [0,evalues,evectors]

SINERTIA=0.4
EPSILON=1.0e-7
def compute_inertia(inertiaflag,radiusflag,dxcom,radius):
    onemass,dx,dy,dz=0,0,0,0
    itensor = np.zeros(6)
    natoms=len(dxcom)
    if (inertiaflag==0) :#{
        inertiaflag = 1;
        for i in range(natoms):
            onemass=1.0
            dx = dxcom[i][0];
            dy = dxcom[i][1];
            dz = dxcom[i][2];
            itensor[0] += onemass * (dy*dy + dz*dz);
            itensor[1] += onemass * (dx*dx + dz*dz);
            itensor[2] += onemass * (dx*dx + dy*dy);
            itensor[3] -= onemass * dy*dz;
            itensor[4] -= onemass * dx*dz;
            itensor[5] -= onemass * dx*dy;
        if (radiusflag==1):
            for i in range(natoms):
                onemass=1.0
                itensor[0] += SINERTIA*onemass * radius[i]*radius[i];
                itensor[1] += SINERTIA*onemass * radius[i]*radius[i];
                itensor[2] += SINERTIA*onemass * radius[i]*radius[i];
    #diagonalize inertia tensor for each body via Jacobi rotations
    #inertia = 3 eigenvalues = principal moments of inertia
    #evectors and exzy = 3 evectors = principal axes of rigid body
    cross = np.zeros(3)
    tensor = np.zeros((3,3))
    evectors= np.zeros((3,3))
    tensor[0][0] = itensor[0];
    tensor[1][1] = itensor[1];
    tensor[2][2] = itensor[2];
    tensor[1][2] = tensor[2][1] = itensor[3];
    tensor[0][2] = tensor[2][0] = itensor[4];
    tensor[0][1] = tensor[1][0] = itensor[5];
    #print (itensor)
    r,inertia,evectors = jacobi(tensor)
    if (r==1):
        print("Insufficient Jacobi rotations for rigid molecule");
    #print ("Jacobi",r,inertia,evectors)
    ex = np.zeros(3)
    ey = np.zeros(3)
    ez = np.zeros(3)
    ex[0] = evectors[0][0];
    ex[1] = evectors[1][0];
    ex[2] = evectors[2][0];
    ey[0] = evectors[0][1];
    ey[1] = evectors[1][1];
    ey[2] = evectors[2][1];
    ez[0] = evectors[0][2];
    ez[1] = evectors[1][2];
    ez[2] = evectors[2][2];
    # if any principal moment < scaled EPSILON, set to 0.0
    qmax = max(inertia[0],inertia[1]);
    qmax = max(qmax,inertia[2]);
    if (inertia[0] < EPSILON*qmax): inertia[0] = 0.0;
    if (inertia[1] < EPSILON*qmax): inertia[1] = 0.0;
    if (inertia[2] < EPSILON*qmax): inertia[2] = 0.0;
    # enforce 3 evectors as a right-handed coordinate system
    # flip 3rd vector if needed
    cross=np.cross(ex,ey)
    if (np.dot(cross,ez)<0.0): ez=-ez
    #quaternion
    #quat = exyz_to_q(ex,ey,ez);
    #print (quat)
    #compute displacements in body frame defined by quat
    return itensor,inertia,ex,ey,ez

    # region rSphereL sphere  0    0  0.0 1250.0   side in
    # region rSphereR sphere  0    0  0.0 1350.0   side out
    # region  rRod  union 2 rSphereL rSphereR  
    # fix fxWall all wall/region rRod harmonic      10.0  0.0  1350.0
    # group mobile subtract all gFixed
def ConvertMolecule(molecule,
                    name,
                    l_mol_defs,
                    l_instances,
                    delta_r,
                    bounds,
                    cname = "",
                    model_data=None):
    """ 
    Convert all information concerning a type of molecule defined in a 
    CellPACK JSON file into moltemplate format.
    In this context, a \"molecule\" is defined as the smallest possible 
    subunit in the JSON files created by CellPACK, and it is assumed to
    contain 'positions', 'radii', and 'results' fields.
    Each molecule_node contains a definition of the type of molecule
    (these details are in the 'name', 'positions', 'radii' fields),
    as well as a list of separate instances (copies) of that molecule
    (in its 'results' field, which specifies the position and orientation
    of every copy of that type of molecule)
    This function converts this information into a moltemplate molecule
    objects, and a list of \"new\" commands which make copies of these
    molecules and move them into the correct positions and orientations.
    """
    name = name.replace(" ","").replace("-","")
    global g_ingredient_id
    #print ("convert molecule "+molecule['name']+" "+name,g_ingredient_id)
    #if (molecule['name'] == "Insulin_crystal"):
    #    return
    instances = []
    if 'results' in molecule:
        instances = molecule['results']
    if len(instances) == 0 and model_data==None:
        return
    l_mol_defs.append(name + ' inherits .../ForceField {\n')
    crds = []
    radii = []
    if not ('positions' in molecule):
        raise InputError('Error: Missing \"positions\" field in \"'+name+'\" molecule.\n')
    if not ('radii' in molecule):
        raise InputError('Error: Missing \"radii\" field in \"'+name+'\" molecule.\n')
    # Typically each molecule's "positions" and "radii" have an additional
    # level of hierachy beneath them. I don't know how many of them there are.
    # Loop over all of these "subunits" and collect all the coordinates
    # from each of them, appending the coordinates together into one big list:
    Xnid = molecule['positions']
    center = [0,0,0]
    for n in range(0, len(Xnid)):  #loop over "subunits" of this molecule
        for i in range(0, int(len(Xnid[n]['coords'])/3)):  #loop over atoms
            crds.append((Xnid[n]['coords'][i*3+0],#left hand input revert on X
                         Xnid[n]['coords'][i*3+1],
                         Xnid[n]['coords'][i*3+2]))
            center = [center[0]+Xnid[n]['coords'][i*3+0],
                      center[1]+Xnid[n]['coords'][i*3+1],
                      center[2]+Xnid[n]['coords'][i*3+2]]
    center = [center[0]/len(crds),center[1]/len(crds),center[2]/len(crds)]
    #inertia = np.dot(np.array(crds).transpose(), np.array(crds))
    #e_values, e_vectors = np.linalg.eig(inertia)
    #cross=np.cross(e_vectors[0],e_vectors[1]);
    #if (np.dot(cross,e_vectors[2]) < 0.0) e_vectors[2]=-e_vectors[2];
    #q=exyz_to_q(e_vectors[0],e_vectors[1],e_vectors[2])
    #print ("#############################")
    ##print (name,center)
    #print (crds)
    #print (e_values, e_vectors)
    #print (q)
    #print ("#############################")
    # Do the same for the radii
    r_ni = molecule['radii']
    for n in range(0, len(r_ni)): #loop over "subunits" of this molecule
        for i in range(0, len(r_ni[n]['radii'])):       #loop over atoms
            radii.append(r_ni[n]['radii'][i])
            #iradius = int(round(radii[i]/delta_r))  #(quantize the radii)
            #ir_needed.add(iradius)
    if len(crds) != len(radii):
        raise InputError('Error: \"positions\" and \"radii\" arrays in \"'+name+'\" have unequal length.\n')
    #itensor1,inertia2,ex,ey,ez=compute_inertia(0,1,crds,radii)
    #print ("#############################")
    #print (radii)
    #print(itensor1)
    #print (inertia2)
    #print (ex)
    #print (ey)
    #print (ez)
    #print (exyz_to_q(ex,ey,ez))
    #print ("#############################")
    l_mol_defs.append('\n'
                      '      #  AtomID MoleculeID AtomType Charge   X    Y    Z\n'
                      '\n')
    l_mol_defs.append('  write(\"Data Atoms\") {\n')
    assert(len(crds) == len(radii))
    list_atom_type = []
    mol = '$mol'
    if (molecule['name'] == "Insulin_crystal"):
        mol='$mol:...'
    for i in range(0, len(crds)):
        iradius = int(round(radii[i]/delta_r))  #(quantize the radii)
        atype_name = '@atom:A' + str(iradius)    #atom type depends on radius
        charge = '0.0'
        l_mol_defs.append('    $atom:a'+str(i+1)+'  '+mol+'  '+atype_name+'  '+charge+'  '+str(crds[i][0])+' '+str(crds[i][1])+' '+str(crds[i][2])+'\n')
        list_atom_type.append(atype_name)
    list_atom_type = set(list_atom_type)    
    N = len(crds)
    group = "g"+cname 
    #group = 'gOrdinary'
    #if N> 1 :
        #group = 'gRigid'  
        #group = "g"+cname  
    #if (molecule['name'] == "Insulin_crystal"): 
    #    group = 'gFixed'        
    list_atom_type_surface = []
    if "surface" in cname :#cname == "surface" :
        #group = 'gSurface'
        #need to add the bicycle if surface
        iradius = int(round(50/delta_r))
        atype_name = '@atom:A' + str(iradius)    #atom type depends on radius
        list_atom_type_surface.append(atype_name)
        charge = '0.0'
        rr=25.0
        l=rr+60.0
        offset = [[-1.732*rr,1.0*rr,0.0*rr],[0.0,-2.0*rr,0.0],[1.732*rr,1.0*rr,0.0]]
        pcp = [0,0,1]#molecule["principalAxis"]
        #offset = molecule["offset"]
        for i in range(3):
            l_mol_defs.append('    $atom:a'+str(N+i+1)+'  '+mol+'  '+atype_name+'  '+charge+'  '+str(pcp[0]*l+offset[i][0])+' '+str(pcp[1]*l+offset[i][1])+' '+str(pcp[2]*l+offset[i][2])+'\n')
        iradius = int(round(55/delta_r))
        atype_name = '@atom:A' + str(iradius)    #atom type depends on radius
        list_atom_type_surface.append(atype_name)
        for i in range(3):
            l_mol_defs.append('    $atom:a'+str(N+i+3+1)+'  '+mol+'  '+atype_name+'  '+charge+'  '+str(pcp[0]*-l+offset[i][0])+' '+str(pcp[1]*-l+offset[i][1])+' '+str(pcp[2]*-l+offset[i][2])+'\n')
    
    l_mol_defs.append('  }  # end of: write(\"Data Atoms\") {...\n\n')
    #grouping ?
    l_mol_defs.append('  write_once(\"In Settings\") {\n')
    l_mol_defs.append('      group '+group+' type '+" ".join(list_atom_type)+'\n')
    if "surface" in cname :#if cname == "surface" :
        l_mol_defs.append('      group gBicycleO type '+list_atom_type_surface[0]+'\n')
        l_mol_defs.append('      group gBicycleI type '+list_atom_type_surface[1]+'\n')
    l_mol_defs.append('}  # end of: write_once(\"In Settings\")\n\n\n')    
    l_mol_defs.append('}  # end of: \"'+name+'\" molecule definition\n\n\n')

    if not ('results' in molecule) and model_data==None :
        raise InputError('Error: Missing \results\" field in \"'+name+'\" molecule.\n')

    deltaXs = []
    quaternions = []
    ninstances = 0
    if (model_data==None):
        instances = molecule['results']
        if len(instances) == 0:
            g_ingredient_id=g_ingredient_id+1
            return
        ninstances = len(instances)
        for i in range(0, ninstances):
            #if not (('0' in instance[i]) and ('1' in instance[i])):
            if not len(instances[i]) == 2:
                raise InputError('Error: Incorrect format in \results\" section of \"'+name+'\" molecule.\n')
            deltaXs.append((-instances[i][0][0],
                            instances[i][0][1],
                            instances[i][0][2]))
            quaternions.append((-instances[i][1][0],instances[i][1][1],instances[i][1][2],-instances[i][1][3]))#in cp we use 1,-1,-1,1 here we tried -1,1,1,-1
            #quaternions.append((0,0,0,1))                    

    else :
        indices = np.nonzero(model_data["pos"][:,3]==g_ingredient_id)
        pos = np.take(model_data["pos"],indices,axis=0)[0]
        if len(pos) == 0:
            g_ingredient_id=g_ingredient_id+1
            return
        ninstances = len(pos)
        rot = np.take(model_data["quat"],indices,axis=0)[0]
        for i in range(0, ninstances):
            #if not (('0' in instance[i]) and ('1' in instance[i])):
            deltaXs.append((-pos[i][0],
                            pos[i][1],
                            pos[i][2]))
            quaternions.append((-rot[i][0],rot[i][1],rot[i][2],-rot[i][3]))#in cp we use 1,-1,-1,1 here we tried -1,1,1,-1
    print (molecule['name'],g_ingredient_id,len(deltaXs))                   
    #l_instances.append('# List of \"'+name+'\" instances:\n')
    if (molecule['name'] == "Insulin_crystal"):
        #later will use the ignredient type for that
        l_instances.append(name+'Body {\n')
        for i in range(0, ninstances):
            l_instances.append(name + '_instances[' + str(i) + '] = new ' + name +
                            '.quat(' + 
                            str(quaternions[i][0]) + ',' +
                            str(quaternions[i][1]) + ',' +
                            str(quaternions[i][2]) + ',' +
                            str(quaternions[i][3]) +
                            ').move(' + 
                            str(deltaXs[i][0]) + ',' +
                            str(deltaXs[i][1]) + ',' +
                            str(deltaXs[i][2]) + ')\n')
        l_instances.append('}\n')
        l_instances.append(name+'_body = new '+name+'Body\n\n')
    else :
        for i in range(0, ninstances):
            l_instances.append(name + '_instances[' + str(i) + '] = new ' + name +
                            '.quat(' + 
                            str(quaternions[i][0]) + ',' +
                            str(quaternions[i][1]) + ',' +
                            str(quaternions[i][2]) + ',' +
                            str(quaternions[i][3]) +
                            ').move(' + 
                            str(deltaXs[i][0]) + ',' +
                            str(deltaXs[i][1]) + ',' +
                            str(deltaXs[i][2]) + ')\n')
    g_ingredient_id=g_ingredient_id+1
    # Now determine the minimum/maximum coordinates of this object
    # and adjust the simulation box boundaries if necessary
    #Xnid = molecule['positions']
    #for n in range(0, len(Xnid)):  #loop over "LOD" of this molecule
    #compute store all intertia and pcp? adjust using first frame ?
    for n in range(0, ninstances):
        for i in range(0, len(crds)):
            X_orig = [crds[i][0],crds[i][1],crds[i][2]]#[0.0, 0.0, 0.0]
            X = [0.0, 0.0, 0.0]
            #for I in range(0, len(Xnid[n]['coords'])/3):  #loop over atoms
            #for d in range(0, 3):
            #    X_orig[d] = Xnid[n]['coords'][I*3+d]
            AffineTransformQ(X, X_orig,
                                    [quaternions[n][0],
                                    quaternions[n][1],
                                    quaternions[n][2],
                                    quaternions[n][3]],
                                    deltaXs[n])
            AdjustBounds(bounds, [X[0]-radii[i], X[1], X[2]])
            AdjustBounds(bounds, [X[0]+radii[i], X[1], X[2]])
            AdjustBounds(bounds, [X[0], X[1]-radii[i], X[2]])
            AdjustBounds(bounds, [X[0], X[1]+radii[i], X[2]])
            AdjustBounds(bounds, [X[0], X[1], X[2]-radii[i]])
            AdjustBounds(bounds, [X[0], X[1], X[2]+radii[i]])
        #m=tr.quaternion_matrix([quaternions[n][3],
        #                            quaternions[n][0],
        #                            quaternions[n][1],
        #                            quaternions[n][2]])
        #new_crds = ApplyMatrix(crds,m)
        #itensor1,inertia2,ex,ey,ez=compute_inertia(0,1,new_crds,radii)
        #print (ex,ey,ez)
        #print (n,inertia2,exyz_to_q(ex,ey,ez))
    #print (bounds)
                #r_ni[n]['radii'][i]

#    N = len(crds)    # = number of particles in this molecule type
#    extra_group = ''
#    if cname == "surface" :
#        extra_group = ( '        group gBicycleO id ' +
#                         '${atom:' + name + '_instances[0]/a'+str(N+1)+'}:' +
#                         '${atom:' + name + '_instances[' +
#                         str(len(instances)-1) + ']/a' +
#                         str(N+3) + '}\n'+
#                        '        group gBicycleI id ' +
#                         '${atom:' + name + '_instances[0]/a'+str(N+4)+'}:' +
#                         '${atom:' + name + '_instances[' +
#                         str(len(instances)-1) + ']/a' +
#                         str(N+6) + '}\n' )
#    if N > 1:
        # Molecules containing multiple atoms use rigid-body integrators to hold
        # each molecule together and preserve its shape.  We must define a group
        # of atoms (gRigid) containing atoms belonging to these rigid molecules.
        # http://lammps.sandia.gov/doc/group.html
        # http://lammps.sandia.gov/doc/fix_rigid.html
#        group = 'gRigid'
#        if cname == "surface" :
#            group = 'gSurface'
#        if (molecule['name'] == "Insulin_crystal"): 
#            group = 'gFixed'
#        l_instances += ['\n',
#                        'write("In Settings") {\n',
#                        ('  group '+group+' id ' +
#                         '${atom:' + name + '_instances[0]/a1}:' +
#                         '${atom:' + name + '_instances[' +
#                         str(len(instances)-1) + ']/a' +
#                         str(N) + '}\n'+extra_group),
#                        '}\n',
#                        '\n',
#                        '\n']
#    else:
        # Molecules containing only one atom do not need to be treated as rigid
        # bodies. We do not need to use rigid body integrators for them. Instead
        # we can use ordinary "fix nve" to integrate their equations of motion.
        # We must define a group of atoms, "gOrdinary", containing these atoms.
        # http://lammps.sandia.gov/doc/group.html
        # http://lammps.sandia.gov/doc/fix_nve.html
#        group = 'gOrdinary'
#        if cname == "surface" :
#            group = 'gSurface'        
#        if (molecule['name'] == "Insulin_crystal"): 
#            group = 'gFixed'
#        l_instances += ['\n',
#                        '\n',
#                        'write("In Settings") {\n',
#                        ('  group '+group+' id ' +
#                         '$atom:' + name + '_instances[0]/a1 \n'+extra_group),
#                        '}\n',
#                        '\n']


    if g_control_vmd_colors:
        for i in range(0, len(instances)):
            l_instances.append('\n')
            l_instances.append('write("vmd_commands.tcl") {  #(optional VMD file)\n')
            for I in range(0, len(radii)):
                r = iradius * delta_r
                atomid_name = '${atom:'+name+'_instances['+str(i)+']/a'+str(I+1)+'}'
                color_name  = '@color:'+name
                l_instances.append('  set sel [atomselect top "index '+atomid_name+'"]\n')
                l_instances.append('  \$sel set name '+color_name+'\n')
            l_instances.append('}  # end of "vmd_commands.tcl"\n')
            l_instances.append('\n')


def ConvertMolecules(molecules,
                     file_out,
                     delta_r,
                     bounds,
                     nindent=0,
                     cname = "",
                     model_data=None):
    #if cname != "surface" :
    #    return
    l_mol_defs = []
    l_instances = []
    for molecule_type_name in molecules:
        ConvertMolecule(molecules[molecule_type_name],
                        molecule_type_name,
                        l_mol_defs,
                        l_instances,
                        delta_r,
                        bounds,cname,model_data=model_data)
        #when debugging, uncomment the next line:
        #break
    #get l_instances from binary file
    file_out.write('\n' + 
                   (nindent*'  ') + '# ----------- molecule definitions -----------\n'
                   '\n')

    file_out.write((nindent*'  ') + (nindent*'  ').join(l_mol_defs))
    file_out.write('\n' +
                   nindent*'  ' + '# ----------- molecule instances -----------\n' +
                   '\n')
    file_out.write((nindent*'  ') + (nindent*'  ').join(l_instances))
    file_out.write('\n')




def ConvertSystem_old(tree,
                  file_out,
                  delta_r,
                  bounds,
                  nindent=0,
                  cname = "",
                  model_data=None):
    """
    Recursively search for objects in the JSON file 
    containing a 'ingredients' field.  
    The 'ingredients' should map to a dictionary of molecule_tyle_names.
    For each of them, define a moltemplate molecule type, followed by
    a list of \"new\" commands which instantiate a copy of that molecule
    at different positions and orientations.
    """

    if not isinstance(tree, dict):
        return
    if 'ingredients' in tree:
        ConvertMolecules(tree['ingredients'],
                         file_out,
                         delta_r,
                         bounds,
                         nindent,cname =cname,model_data=model_data)
    else:
        for object_name in tree:
            if object_name == 'recipe':
                continue
            file_out.write(nindent*'  '+object_name + ' {\n')
            if object_name == 'cytoplasme':
                ConvertMolecules(tree['cytoplasme'],
                                file_out,
                                delta_r,
                                bounds,
                                nindent,cname =cname,model_data=model_data)
            else :                    
                ConvertSystem(tree[object_name],
                            file_out,
                            delta_r,
                            bounds,
                            nindent+1,
                            cname=object_name,model_data=model_data)
            file_out.write(nindent*'  '+'}  # endo of \"'+object_name+'\" definition\n\n')
            file_out.write('\n' + 
                           nindent*'  '+object_name + '_instance = new ' + object_name + '\n' +
                           '\n' +
                           '\n')

            #when debugging, uncomment the next line:
            #break

def ConvertSystem(tree,
                  file_out,
                  delta_r,
                  bounds,
                  nindent=0,
                  cname = "",
                  model_data=None):
    if not isinstance(tree, dict):
        return
    if 'cytoplasme' in tree and len(tree['cytoplasme']):
        object_name = 'cytoplasme'
        file_out.write(nindent*'  cytoplasme {\n')
        ConvertMolecules(tree['cytoplasme']['ingredients'],
                file_out,
                delta_r,
                bounds,
                nindent,cname ='cytoplasme',model_data=model_data)
        file_out.write(nindent*'  '+'}  # endo of \"'+object_name+'\" definition\n\n')
        file_out.write('\n' + 
                           nindent*'  '+object_name + '_instance = new ' + object_name + '\n' +
                           '\n' +
                           '\n')
    if 'compartments' in tree:
        for compname in tree['compartments']:
            if 'surface' in tree['compartments'][compname]:
                #list_groups.append("g"+compname+"_surface")
                cname = object_name = compname+"_surface"
                file_out.write(nindent*'  '+object_name + ' {\n')
                ConvertMolecules(tree['compartments'][compname]['surface']['ingredients'],
                         file_out,
                         delta_r,
                         bounds,
                         nindent,cname =cname,model_data=model_data)
                file_out.write(nindent*'  '+'}  # endo of \"'+object_name+'\" definition\n\n')
                file_out.write('\n' + 
                                nindent*'  '+object_name + '_instance = new ' + object_name + '\n' +
                                '\n' +
                                '\n')                         
            if 'interior' in tree['compartments'][compname]:
                #list_groups.append("g"+compname+"_interior")
                cname = object_name = compname+"_interior"
                file_out.write(nindent*'  '+object_name + ' {\n')
                ConvertMolecules(tree['compartments'][compname]['interior']['ingredients'],
                         file_out,
                         delta_r,
                         bounds,
                         nindent,cname =cname,model_data=model_data)
                file_out.write(nindent*'  '+'}  # endo of \"'+object_name+'\" definition\n\n')
                file_out.write('\n' + 
                                nindent*'  '+object_name + '_instance = new ' + object_name + '\n' +
                                '\n' +
                                '\n')      

def CreateGroupCompartment(tree):
    list_groups=[]
    if not isinstance(tree, dict):
        return
    #start with cytoplasme, then compartment
    if 'cytoplasme' in tree and len(tree['cytoplasme']):
        list_groups.append("gcytoplasme")
    if 'compartments' in tree:
        for compname in tree['compartments']:
            if 'surface' in tree['compartments'][compname]:
                list_groups.append("g"+compname+"_surface")
            if 'interior' in tree['compartments'][compname]:
                list_groups.append("g"+compname+"_interior")
    return list_groups

#f="C:\\Users\\ludov\\Documents\\BrettISG\\models_oct13\\cf1_model0_1_0.bin"
#import struct
#f=open(model_in,"rb")
def GetModelData(model_in):
    import struct
    f=open(model_in,"rb")
    ninst=struct.unpack('<i', f.read(4))[0]
    ncurve=struct.unpack('<i', f.read(4))[0]
    #ninstb=f.read(4)
    #ncurveb=f.read(4)
    data=f.read(ninst*4*4) 
    pos = np.frombuffer(data,dtype='f').reshape((ninst,4))
    data=f.read(ninst*4*4) 
    quat = np.frombuffer(data,dtype='f').reshape((ninst,4))
    #point,norm,info curve
    if  ncurve!= 0:
        data=f.read(ncurve*4*4) 
        pts = np.frombuffer(data)
        #data=f.read(ninst*4*4) 
        #quat = np.frombuffer(data)  
    f.close()   
    return {"pos":pos,"quat":quat}

    #pxyzID
def ConvertCellPACK(file_in,        # typically sys.stdin
                    filename_in,    # optional file name
                    file_out,       # typically sys.stdout
                    filename_out,   # optional file name
                    model_in,       # optional model file name
                    out_obj_name,   # optional moltemplate object name
                    delta_r,        # radial resolution
                    pairstyle,      # which LAMMPS pair_style do you want?
                    pairstyle2docs, # documentation for these pair styles
                    pairstyle2args, # required arguments for these pair styles
                    epsilon,        # Lennard-Jones parameter
                    debye):         # Debyle length (if applicable)
    """
    Read a JSON file created by CellPACK and 
    convert it to MOLTEMPLATE (LT) format.
    """
    tree = json.load(file_in,object_pairs_hook=OrderedDict)#retain file order
    model_data=None
    if (model_in!=""):
        #gather array of pos/quat
        model_data =  GetModelData(model_in)
    print (model_in,model_data)
    # depth-first-search the JSON tree looking
    # for any nodes containing the string 'ingredients'
    # convert them to molecule definitions and
    # lists of instances (copies)
    if out_obj_name != '':
        file_out.write('# This file defines a type of molecular object (\"'+out_obj_name+'\") contaning\n'
                       '# the entire system from the JSON output of CellPACK.  Later on, before you\n'
                       '# use moltemplate.sh, you must create a file (eg \"system.lt\") containing:\n'
                       '# import \"'+filename_out+'\"\n'
                       '# system = new '+out_obj_name+'\n'
                       '#  OR, if you want multiple copies of your system, replace the line above with:\n'
                       '# copy_1 = new '+out_obj_name+'.rot(theta1,ax1,ay1,az1).move(x1,y1,z1)\n'
                       '# copy_2 = new '+out_obj_name+'.rot(theta2,ax2,ay2,az2).move(x2,y2,z2)\n'
                       '# copy_3 = new '+out_obj_name+'.rot(theta3,ax3,ay3,az3).move(x3,y3,z3)\n'
                       '#     :\n'
                       '# Then run moltemplate.sh on the \"system.lt\" file using:\n'
                       '#\n'
                       '# moltemplate.sh -nocheck '+filename_out+'\n')
                       #'#      (\"moltemplate.sh system.lt\" works too but takes longer)\n')
    else:
        file_out.write('# Later, you can run moltemplate on this file using\n'
                       '# moltemplate.sh -nocheck '+filename_out+'\n')
                       #'#       (moltemplate.sh '+filename_out+' works too but takes longer)\n')
    #file_out.write('# On a large system, moltemplate.sh can up to 2 hours to complete.\n'





    file_out.write('\n'
                   '\n')
    nindent = 0
    if out_obj_name != '':
        file_out.write(out_obj_name + ' {\n')
        nindent = 1


    file_out.write('\n\n'
                   'ForceField {\n'
                   '\n'
                   '   # The \"ForceField\" object defines a list of atom types\n'
                   '   # and the forces between them.  These atom types and forces\n'
                   '   # are shared by all of the molecule types in this simulation.\n'
                   '\n')




    # Figure out what kinds of particles we need
    # These will be referenced below when we
    # create a "ForceField" object which describes them.

    ir_needed = set([]) # Which particle radii are needed in this simulation?
                        # LAMMPS does not allow every particle in the sim to
                        # have a unique radius.  The number of particle types
                        # cannot exceed roughly 10^3 (for efficient execution)
                        # Hence, the radii of each particle is quantized by
                        # rounding it to the nearest multiple of "delta_r".
                        # All particles whose radii fall into this bin
                        # are represented by the same particle type.
                        # The "ir_needed" keeps track of which bins have been
                        # visited.  Later, we will loop through all the bins
                        # in "ir_needed", and define particle types for the
                        # corresponding particles that fall into that bin
                        # (that range of radii sizes), and choose appropriate
                        # force-field parameters (Lennard-Jones parameters)
                        # for those particle types.

    RadiiNeeded(tree, ir_needed, delta_r)
    #add the bycycle
    iradius = int(round(50.0/delta_r))  #(quantize the radii)
    ir_needed.add(iradius)
    iradius = int(round(55.0/delta_r))  #(quantize the radii)
    ir_needed.add(iradius)
    assert(len(ir_needed) > 0)

    rmax = max(ir_needed) * delta_r
    rcut_max = rmax
    bounds = [[0.0,0.0,0.0],    #Box big enough to enclose all the particles
              [-1.0,-1.0,-1.0]] #[[xmin,ymin,zmin],[xmax,ymax,zmax]]
    pairstyle2args['lj/cut'] = str(2*rcut_max)
    pairstyle2args['lj/cut/coul/debye'] = str(debye)+' '+str(rcut_max)+' '+str(rcut_max)
    pairstyle2args['lj/cut/coul/cut'] = str(rcut_max)
    pairstyle2args['lj/cut/coul/long'] = str(rcut_max)
    pairstyle2args['lj/class2/coul/long'] = str(rcut_max)
    pairstyle2args['lj/class2/coul/cut'] = str(rcut_max)
    pairstyle2args['gauss'] = str(rcut_max)
    #pair_mixing_style = 'geometric' <-- NO do not use geometric
    pair_mixing_style = 'arithmetic'
    special_bonds_command = 'special_bonds lj/coul 0.0 0.0 1.0'

    file_out.write('   write_once("In Settings Pair Coeffs") {\n')
    for iradius in sorted(ir_needed):
        rcut = 2 * iradius * delta_r   #(don't forget the 2)
        r = rcut / (2.0**(1.0/6))
        file_out.write('     pair_coeff ' +
                       '@atom:A' + str(iradius) + ' ' +
                       '@atom:A' + str(iradius) + ' ' +
                       pairstyle + ' ' +
                       str(epsilon) + ' ' +
                       str(r) + ' ' +
                       str(rcut) + '\n')
    file_out.write('  }  #end of "In Settings Pair Coeffs"\n'
                   '\n\n\n')
    default_mass = 1.0   # Users can override this later



    file_out.write('   # Alternate forces used in the initial stages of minimization:\n\n')
    file_out.write('   write_once("In Settings Pair Coeffs Soft") {\n')
    for iradius in sorted(ir_needed):
        rcut = 2 * iradius * delta_r   #(don't forget the 2)
        # pair_style gauss uses the following formula for the energy:
        # Upair(r) = A*exp(-B*r**2) = A*exp(-(0.5)*(r/sigma)**2)
        # I will set the force distance cutoff ("rcut") to the radius
        # defined by the JSON file created by CellPACK.
        # The force between particles separated by larger than this distance
        # is zero.  (We don't want the system to swell during minimization.)
        # I'll choose the sigma parameter (gaussian width) so that the gaussian
        # is half its peak height at r=rcut: <-> exp(-(1/2)(rcut/sigma)**2)=0.5
        sigma = rcut / sqrt((log(2)*2))
        # Then I will double the height of the Gaussian to compensate:
        A = epsilon * 2.0
        # The arguments for the pair_coeff command are "A", "B", and "rcut"
        B = 0.5*(1.0/(sigma**2))
        file_out.write('     pair_coeff ' +
                       '@atom:A' + str(iradius) + ' ' +
                       '@atom:A' + str(iradius) + ' ' +
                       'gauss' + ' ' +
                       str(-A) + ' ' +
                       str(B) + ' ' +
                       str(rcut) + '\n')
    file_out.write('  }  #end of "In Settings Pair Coeffs Soft"\n'
                   '\n\n\n')
    default_mass = 1.0   # Users can override this later



    
    file_out.write('  # Last I checked, LAMMPS still requires that every atom has its\n'
                   '  # mass defined in the DATA file, even if it is irrelevant.\n'
                   '  # Take care of that detail below.\n')
    file_out.write('\n'
                   '   write_once("Data Masses") {\n')

    for iradius in sorted(ir_needed):
        #rcut = iradius * delta_r
        #r = rcut / (2.0**(1.0/6))
        file_out.write('    ' +
                       '@atom:A' + str(iradius) +' '+ str(default_mass) +'\n')
    file_out.write('  }  # end of "Data Masses"\n\n')


    file_out.write('\n\n'
                   '  # At some point we must specify what -kind- of force fields we want to use\n'
                   '  # We must also specify how we want to measure distances and energies (units)\n'
                   '  # as well as how to represent the particles in the simulation (atom_style)\n'
                   '\n'
                   '  write_once("In Init") {\n'
                   '\n'
                   '    atom_style full #(default atom style)\n'
                   '\n'
                   '    units lj        #(this means the units can be customized by the user (us))\n'
                   '\n'
                   '    pair_style hybrid ' + pairstyle + ' ' +
                   pairstyle2args[pairstyle] + '\n\n\n')
    
    
    file_out.write('    # For details, see\n' +
                   '    # ' + pairstyle2docs[pairstyle] + '\n' +
                   '    # ' + pairstyle2docs['gauss'] + '\n')

    file_out.write('\n'
                   '    # The next command is optional but useful when using Lennard Jones forces:\n'
                   '\n'
                   '    pair_modify shift yes\n'
                   '\n'
                   '\n')

    file_out.write('\n'
                   '    # Use ordinary Lorenz-Berthelot mixing rules.\n'
                   '    # (This means the effective minimal distance\n'
                   '    #  between unlike particles is the sum of their radii,\n'
                   '    #  as it would be if they were hard spheres)\n'
                   '\n'
                   '    pair_modify mix ' + pair_mixing_style + '\n'
                   '\n')

    file_out.write('\n'
                   '    # The next line is probably not relevant, since we\n'
                   '    # do not have any bonds in our system.\n'
                   '    ' + special_bonds_command + '\n'
                   '\n')

    file_out.write('\n'
                   '    # If I am not mistaken, the next two lines help speed up the computation\n'
                   '    # of neighbor lists.  They are necessary because of the wide diversity\n'
                   '    # of particle sizes in the simulation.  (Most molecular dynamics software\n'
                   '    # was optimized for particles of similar size.)\n'
                   '\n'
                   '    comm_modify mode multi  # Needed since we have many different particle sizes\n'
                   '    neighbor 3.0 multi      # Adjust this number later to improve efficiency\n'
                   '\n')
    file_out.write('  }  # finished selecting force field styles\n')

    file_out.write('  # specify the custom force field style used for minimization\n'
                   '  write_once("In Init Soft") {\n'
                   '    pair_style hybrid gauss ' + 
                   pairstyle2args['gauss'] + '\n'
                   '  }\n\n\n')

    file_out.write('\n'
                   '\n'
                   '  # Optional: Create a file to help make display in VMD prettier\n'
                   '\n'
                   '  write_once("vmd_commands.tcl") {\n')
    #is it important to sort them ?
    for iradius in sorted(ir_needed):
        r = iradius * delta_r
        atype_name = '@{atom:A' + str(iradius) + '}'
        file_out.write('    set sel [atomselect top "type '+atype_name+'"]\n'
                       '    \$sel set radius '+str(r)+'\n')
        if iradius == 500:
            file_out.write('    mol rep VDW\n'
                '    mol addrep 0\n'
                '    mol modselect 1 0 "type '+atype_name+'"\n'
                '    mol modcolor 1 0 "ColorID 0"\n')
        if iradius == 550:
            file_out.write('    mol rep VDW\n'
                '    mol addrep 0\n'
                '    mol modselect 2 0 "type '+atype_name+'"\n'
                '    mol modcolor 2 0 "ColorID 1"\n')            
    atype_name1 = '@{atom:A500}'          
    atype_name2 = '@{atom:A550}'    
    file_out.write(' mol rep VDW\n')
    file_out.write(' mol addrep 0\n')
    file_out.write(' mol modselect 3 0 "not type '+atype_name1+' '+atype_name2+'"\n')
    #file_out.write(' mol modcolor 3 0 "ColorID 2"\n')
    file_out.write(' draw sphere \{0 0 0\} radius 1600.0 resolution 20\n') 
    file_out.write(' draw material Transparent\n') 
    file_out.write('  }  # end of "vmd_commands.tcl"\n\n')

    #need one outside group
    #need one group per compartment
    list_groups = CreateGroupCompartment(tree)
    group_string=""
    for g in list_groups:
        group_string+="    group "+g+" id 1       # define a group\n"
        group_string+="    group "+g+" clear       # initialize it to be empty\n"
    file_out.write('\n'
                   '\n'
                   '  write_once("In Settings") {\n' +
                   '\n'
                   '    # Note: Atoms in the "gRigid" group use a rigid-body integrator to keep\n'
                   '    #       them from moving relative to eachother in each molecule.\n'
                   '    #       Other atoms (in the "gOrdinary") group use the standard Verlet\n'
                   '    #       algorithm for movement. In some cases these groups are empty.\n'
                   '    #       Even so, we must make sure both the "gRigid" and "gOrdinary"\n'
                   '    #       groups are defined so we can refer to them later. We do that below\n'
                   '    # http://lammps.sandia.gov/doc/group.html\n'+
                   group_string +
                   #'    group gRigid id 1       # define a group\n' +
                   #'    group gRigid clear      # initialize it to be empty\n' +
                   '    group gOrdinary id 1    # define a group\n' +
                   '    group gOrdinary clear   # initialize it to be empty\n' +
                   '    group gFixed id 1    # define a group\n' +
                   '    group gFixed clear   # initialize it to be empty\n' +     
                   '    group gBicycleI id 1    # define a group\n' +
                   '    group gBicycleI clear   # initialize it to be empty\n' +          
                   '    group gBicycleO id 1    # define a group\n' +
                   '    group gBicycleO clear   # initialize it to be empty\n' +                                                
                   #'    group gSurface id 1    # define a group\n' +
                   #'    group gSurface clear   # initialize it to be empty\n' +                    
                   '    #\n'
                   '    # Note: Later on we will use:\n'
                   '    #   fix fxRigid gRigid rigid molecule\n'
                   '    #   fix fxNVE gOrdinary nve\n'
                   '    # For details see:\n'
                   '    #   http://lammps.sandia.gov/doc/fix_rigid.html\n'
                   '    #   http://lammps.sandia.gov/doc/fix_nve.html\n'
                   '    #\n'
                   '    #region rSphereI sphere  0    0  0  1600  side out\n'
                   '    #region rSphereO sphere  0    0  0  1600  side in\n'
                   '    # neigh_modify one 20000 page 200000\n'
                   '    neigh_modify exclude molecule/intra all\n'
                   '    neigh_modify exclude type '+atype_name1+' '+atype_name2+'\n'
                   '    neigh_modify exclude molecule/inter gBicycleI\n'
                   '    neigh_modify exclude molecule/inter gBicycleO\n'
                   '  }\n'
                   '  ### This next line greatly increases the speed of the simulation:\n'
                   '  ### No need to calculate forces between particles in the same rigid molecule\n'
                   '  # neigh_modify exclude molecule/intra gRigid\n'
                   '  # http://lammps.sandia.gov/doc/neigh_modify.html\n'
                   #'  #}\n'
                   '\n')
    file_out.write('}  # end of the "ForceField" object definition\n'
                   '\n\n\n\n\n')
    

    ConvertSystem(tree,
                  file_out,
                  delta_r,
                  bounds,
                  nindent,
                  model_data=model_data)

    if out_obj_name != '':
        file_out.write('}  # end of \"'+ out_obj_name + '\" object definition\n')

    

    # Print the simulation boundary conditions
    print (bounds)
    file_out.write('\n\n'
                   '# Simulation boundaries:\n'
                   '\n'
                   'write_once("Data Boundary") {\n'
                   '  '+str(bounds[0][0])+' '+str(bounds[1][0])+' xlo xhi\n'
                   '  '+str(bounds[0][1])+' '+str(bounds[1][1])+' ylo yhi\n'
                   '  '+str(bounds[0][2])+' '+str(bounds[1][2])+' zlo zhi\n'
                   '}\n\n')








def main():
    try:
        sys.stderr.write(g_program_name + ", version " +
                         __version__ + ", " + __date__ + "\n")

   
        if sys.version > '3':
            import io
        else:
            import cStringIO
    
        # defaults:
        out_obj_name = ''
        type_subset = set([])

        filename_in = ''
        filename_out = 'THIS_FILE'
        file_in = sys.stdin
        file_out = sys.stdout
        model_in = ""
        delta_r = 0.1     # resolution of particle radii

        # --- Units ---
        # The following parameters depend on what units you are using.
        # By default, this program assumes distances are in nm,
        # energies are in kCal/Mole,
        # Note: When running LAMMPS, temperatures in the fix_langevin or
        #       fix_nvt commands must be specified by the user in units of
        #       energy (ie kCal/mole), not K.   (In these units,
        #       k_B*temperature = 0.5961621, assuming temperature = 300K)
        debye = 1.0       # the Debye length(only relevent for some pair_styles)
        kB = 0.001987207    # Default Boltzmann's constant ((kCal/Mole)/degreeK)
        temperature = 300.0 # Default temperature (in K)
        epsilon = kB*temperature  # The Lennard-Jones "epsilon" parameter has
                                  # units of energy and should be approximately
                                  # equal to the value of k_B*temperature in 
                                  # whatever units you are using.
        pairstyle = 'lj/cut'
        pairstyle2docs = {}
        pairstyle2args = defaultdict(str)
        pairstyle2docs['lj/cut'] = 'http://lammps.sandia.gov/doc/pair_lj.html'
        pairstyle2docs['lj/cut/coul/debye'] = 'http://lammps.sandia.gov/doc/pair_lj.html'
        pairstyle2docs['lj/cut/coul/cut'] = 'http://lammps.sandia.gov/doc/pair_lj.html'
        pairstyle2docs['lj/cut/coul/long'] = 'http://lammps.sandia.gov/doc/pair_lj.html'
        pairstyle2docs['lj/class2/coul/long'] = 'http://lammps.sandia.gov/doc/pair_class2.html'
        pairstyle2docs['lj/class2/coul/cut'] = 'http://lammps.sandia.gov/doc/pair_class2.html'
        pairstyle2docs['gauss'] = 'http://lammps.sandia.gov/doc/pair_gauss.html'

        argv = [arg for arg in sys.argv]
    
        i = 1
    
        while i < len(argv):
    
            if argv[i] == '-name':
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by the name of the\n'
                                     '       moltemplate object you wish to create.\n')
                out_obj_name = argv[i + 1]
                del argv[i:i + 2]

            elif argv[i] == '-pairstyle':
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by a LAMMPS pair_style.\n')
                pairstyle = argv[i + 1]
                if not pairstyle in pairstyle2docs:
                    raise InputError('Error: Invalid '+argv[i]+' argument.  Available choices are:\n'
                                     '\n'.join([ps for ps in pairstyle2docs])+'\n')
                del argv[i:i + 2]

            elif argv[i] in ('-deltaR', '-deltar', '-delta-r'):
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by a number.\n')
                delta_r = float(argv[i + 1])
                del argv[i:i + 2]

            elif argv[i] == '-epsilon':
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by a number.\n')
                epsilon = float(argv[i + 1])
                del argv[i:i + 2]

            elif argv[i] == '-debye':
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by a number.\n')
                debye = float(argv[i + 1])
                del argv[i:i + 2]

            elif argv[i] in ('-in', '-file'):
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by\n'
                                     '       the name of a JSON file created by CellPACK.\n')
                filename_in = argv[i + 1]
                try:
                    file_in = open(filename_in, 'r')
                except IOError:
                    sys.stderr.write('Error: Unable to open file\n'
                                     '       \"' + filename_in + '\"\n'
                                     '       for reading.\n')
                    sys.exit(1)
                del argv[i:i + 2]

            elif argv[i] in ('-out'):
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by the name of a moltemplate file\n')
                filename_out = argv[i + 1]
                try:
                    file_out = open(filename_out, 'w')
                except IOError:
                    sys.stderr.write('Error: Unable to open file\n'
                                     '       \"' + filename_out + '\"\n'
                                     '       for writing.\n')
                    sys.exit(1)
                del argv[i:i + 2]
            elif argv[i] =='-model':
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by the name of a file\n')
                model_in= argv[i + 1]
                del argv[i:i + 2]
            elif argv[i] in ('-url', '-in-url'):
                import urllib2
                if i + 1 >= len(argv):
                    raise InputError('Error: ' + argv[i] + ' flag should be followed by a URL\n'
                                    '       pointing to a JSON file created by CellPACK.\n')

                url = argv[i + 1]
                try:
                    request = urllib2.Request(url)
                    file_in = urllib2.urlopen(request)
                except urllib2.URLError:
                    sys.stdout.write("Error: Unable to open link:\n" + url + "\n")
                    sys.exit(1)
                del argv[i:i + 2]
    
            elif argv[i] in ('-help', '--help', '-?', '--?'):
                sys.stderr.write(doc_msg)
                sys.exit(0)
                del argv[i:i + 1]
    
            else:
                i += 1
    
        if len(argv) != 1:
            raise InputError('Error: Unrecongized arguments: ' + ' '.join(argv[1:]) +
                             '\n\n' + doc_msg)



        file_out.write('# This file was generated automatically using:\n'+
                       '#    '+g_program_name+' '+' '.join(sys.argv[1:])+'\n')


        ConvertCellPACK(file_in,
                        filename_in,
                        file_out,
                        filename_out,
                        model_in,                        
                        out_obj_name,
                        delta_r,
                        pairstyle,
                        pairstyle2docs,
                        pairstyle2args,
                        epsilon,
                        debye)
        return file_in


    except InputError as err:
        sys.stderr.write('\n\n' + str(err) + '\n')
        sys.exit(1)

if __name__ == '__main__':
    file_in=main()
# need a minimum of 2 beads per rigid body or ellipsoid ? 
#python  ~/Documents/cellpack2moltemplate/cellpack2moltemplate/cellpack2lt.py -in resultsISGtest.json  -out system2.lt
#sh ~/Documents/moltemplate/moltemplate/scripts/moltemplate.sh system.lt -nocheck#nocheck?
#"C:\Program Files\LAMMPS 64-bit 23Oct2017\bin\lmp_serial.exe" -i run.in.min
#vmd traj_min.lammpstrj -e vmd_commands.tcl
#python  ~/Documents/cellpack2moltemplate/cellpack2moltemplate/cellpack2lt.py -in result.json  -out system.lt;sh ~/Documents/moltemplate/moltemplate/scripts/moltemplate.sh system.lt;"C:\Program Files\LAMMPS 64-bit 23Oct2017\bin\lmp_serial.exe" -i run.in.min1;vmd traj_min_soft.lammpstrj -e vmd_commands.tcl
#omp lammps : 
#"C:\Program Files\LAMMPS 64-bit 23Oct2017\bin\lmp_serial.exe" -sf omp -pk omp 16 -i run.in.min1
#gpu -sf gpu -pk gpu 1 doesnt work on my windowsmachine
#write_once("In Settings") {
#    group gSurface type @atom:A661 @atom:A497 @atom:A756 @atom:A749
#    group gBicycleI type @atom:A500
#    group gBicycleO type @atom:A550
#}
#python -i  ~/Documents/cellpack2moltemplate.git/cellpack2moltemplate/cellpack2lt.py -in models_oct13.json  -out system.lt -model cf1_model0_1_0.bin
#python -i  C:\Users\ludov\Documents\cellpack2moltemplate.git\cellpack2moltemplate\cellpack2lt_new.py -in C:\Users\ludov\Documents\ISG\models\recipes\initialISG.json -out C:\Users\ludov\Documents\ISG\models\model_systematic\models\relaxed\system.lt -model C:\Users\ludov\Documents\ISG\models\model_systematic\models\mature\cfx_model0_0_0.bin
#sh ~/Documents/moltemplate/moltemplate/scripts/moltemplate.sh system.lt -nocheck #1h30 later
#1175640 rigid bodies with 3224998 atoms ISG
##"C:\Program Files\LAMMPS 64-bit 19Sep2019\bin\lmp_serial.exe" -sf omp -pk omp 16 -i run.in.min1
#"C:\Program Files\LAMMPS 64-bit 19Sep2019-MPI\bin\lmp_mpi.exe" -sf omp -pk omp 16 -i run.in.min1
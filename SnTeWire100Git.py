import numpy as np
from numpy import save
import cmath
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use('Agg')
import sys

nb = int(sys.argv[1]) # number of atoms

nfi = float(sys.argv[2]) # number of flux quanta through the wire


# Defining i and pi
Pi = np.pi
I = 1j

# Defining Pauli matrices, identity matrices and orbital vectors
id2 = np.identity(2)
id3 = np.identity(3)

# sig=spin subspace
sig1 = np.array([[0, 1.], [1., 0]])
sig2 = np.array([[0, -I], [I, 0]])
sig3 = np.array([[1., 0], [0, -1.]])
sig=np.array([sig1, sig2, sig3])

# tau=sublattice subspace
tau1 = np.array([[0, 1.], [1., 0]])
tau2 = np.array([[0, -I], [I, 0]])
tau3 = np.array([[1., 0], [0, -1.]])

# Orbital vectors
orb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
orb1= np.array([[[[1,0,0],[0,0,0],[0,0,0]],[[0,1,0],[0,0,0],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]]]
               ,[[[0,0,0],[1,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,1],[0,0,0]]],
                [[[0,0,0],[0,0,0],[1,0,0]],[[0,0,0],[0,0,0],[0,1,0]],[[0,0,0],[0,0,0],[0,0,1]]]])

# Vectors to n.n., different sublattice
a =np.array( [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
a1=np.array([[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],
             [[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]]])

# Vectors to n.n.n., same sublattice
d = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1],
              [1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]])
d1=np.array([[[1,1,0],[1,1,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,1,1],[0,1,1]],
             [[1,1,0],[1,1,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,1,1],[0,1,1]],
             [[1,-1,0],[-1,1,0],[0,0,0]],[[1,0,-1],[0,0,0],[-1,0,1]],[[0,0,0],[0,1,-1],[0,-1,1]],
             [[1,-1,0],[-1,1,0],[0,0,0]],[[1,0,-1],[0,0,0],[-1,0,1]],[[0,0,0],[0,1,-1],[0,-1,1]]])


def delta(i , j):
    if i == j:
        return 1
    else:
        return 0

# WIRE HAMILTONIAN WITH MAGNETIC FIELD

def hribb(rx, ry, kz, n):
    fi=complex(nfi/((nb-1)**2))
    m = complex(1.65)
    t11 = complex(0.5)
    t12 = complex(0.9)
    t21 = complex(0.9)
    Lam1 = complex(0.7)
    Lam2 = complex(0.7)
    hr=m * delta(rx, 0) * delta(ry, 0) * np.kron(id2, np.kron(tau3, id3))
    for j in range(0, len(a)):
        hr += t12 * delta(rx, a[j, 0]) * delta(ry, a[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], a[j])) * cmath.exp(I * 2 * Pi * n * fi * ry) * np.kron(id2, np.kron((tau1 - I * tau2) * 0.5, a1[j]))
        hr += t21 * delta(rx, a[j, 0]) * delta(ry, a[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], a[j])) * cmath.exp(I * 2 * Pi * n * fi * ry) * np.kron(id2, np.kron((tau1 + I * tau2) * 0.5, a1[j]))
    for j in range(0, len(d)):
        hr += t11 * delta(rx, d[j, 0]) * delta(ry, d[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], d[j])) * cmath.exp(I * 2 * Pi * (n * fi * ry + 0.5 * fi * ry * rx)) * np.kron(id2, np.kron(tau3, d1[j] * 0.5))
    for k in range(0, 3):
        for j in range(0, 3):
            hr += -I * Lam1 * delta(rx, 0) * delta(ry, 0) * np.kron(
                (np.cross(orb[k], orb[j])[0] * sig[0] + np.cross(orb[k], orb[j])[1] * sig[1] + np.cross(orb[k], orb[j])[2] * sig[2])
                , np.kron((id2 + tau3) * 0.5, orb1[k, j]))
            hr += -I * Lam2 * delta(rx, 0) * delta(ry, 0) * np.kron(
                (np.cross(orb[k], orb[j])[0] * sig[0] + np.cross(orb[k], orb[j])[1] * sig[1] + np.cross(orb[k], orb[j])[2] * sig[2])
                , np.kron((id2 - tau3) * 0.5, orb1[k, j]))
    return hr


def hwire(kz):

    # Matrix on the diagonal : on site and  hopping in X direction
    hwx = np.kron(np.identity(nb), hribb(0, 0, kz, 0)) + np.kron(np.eye(nb, k=-1), hribb(1, 0, kz, 0)) + np.kron(np.eye(nb, k=1), hribb(-1, 0, kz, 0))

    # Matrix on the - 1 off - diagonal - hopping in Y direction and in XY directions
    hwym = np.zeros((12 * nb, 12 * nb), dtype=complex)
    for j in range(0, nb):
        z1 = np.zeros((nb, nb), dtype=complex)
        z1[j, j] = complex(1.0)
        hwym += np.kron(z1, hribb(0, 1, kz, j+1))
    for k in range(0, nb-1):
        z2 = np.zeros((nb, nb), dtype=complex)
        z2[k+1, k] = complex(1.0)
        z3 = np.zeros((nb, nb), dtype=complex)
        z3[k, k+1] = complex(1.0)
        hwym += np.kron(z2, hribb(1, 1, kz, k+1)) + np.kron(z3, hribb(-1, 1, kz, k+2))

    # Matrix on the 1 off - diagonal - hopping in -Y direction and in XY directions
    hwyp = np.zeros((12 * nb, 12 * nb), dtype=complex)
    for j in range(0, nb):
        z1 = np.zeros((nb, nb), dtype=complex)
        z1[j, j] = complex(1.0)
        hwyp += np.kron(z1, hribb(0, -1, kz, j+1))
    for k in range(0, nb-1):
        z2 = np.zeros((nb, nb), dtype=complex)
        z2[k+1, k] = complex(1.0)
        z3 = np.zeros((nb, nb), dtype=complex)
        z3[k, k+1] = complex(1.0)
        hwyp += np.kron(z2, hribb(1, -1, kz, k+1)) + np.kron(z3, hribb(-1, -1, kz, k+2))

    # Final wire Hamiltonian:
    hw = np.kron(np.identity(nb), hwx) + np.kron(np.eye(nb, k=-1), hwym) + np.kron(np.eye(nb, k=1), hwyp)
    hws=scipy.sparse.csr_matrix(hw)
    return hws


# Creating k values for wire band structure calculations
ki = float(sys.argv[3])*Pi # left limit of the spectrum
kf = float(sys.argv[4])*Pi # right limit of the spectrum
kw_points = 101 # number of k points per calculation
divide_k = int(sys.argv[5]) # the value in which to increas density of k points
divide_k_index = int(sys.argv[6]) # numbering the sub calculations to get complex band structure
kw_val = np.linspace(ki, kf, kw_points)

def offset(index):
    return np.float64((kf-ki)*(index-1)/((kw_points-1)*divide_k))
#print(offset(divide_k_index)/Pi)

# Creating the wire band structure data: an array of the eigenvalues at all k-points
num_ev = 30
wbands = np.zeros((kw_points, num_ev), dtype=float)
#wfunctions = np.zeros((kw_points, 12*nb*nb, num_ev),dtype=complex) #wave functions: for each k, 120 w.f. of size 12nb^2

for i in range(0, kw_points):
    w, v = scipy.sparse.linalg.eigsh(hwire(kw_val[i]+offset(divide_k_index)), num_ev, sigma=-0.1)
    wbands[i]=w
    # wfunctions[i]=v


file_name='CLSR_'+str(nb)+'L_'+str(num_ev)+'ev_'+str(kw_points)+'kp_'+str(divide_k_index)+'_of_'+str(divide_k)+'_'+str(nfi)+'fi'

save(file_name+'.npy', wbands)
# save(file_name+'wavefun.npy',wfunctions)



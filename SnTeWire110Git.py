import numpy as np
from numpy import save
import math
import cmath
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
matplotlib.use('Agg')
import sys

# Defining i, pi and Sqrt(2)
Pi = np.pi
I = 1j
Sqrt2=math.sqrt(2)

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
a45=np.array( [[1/(2*Sqrt2), 1/(2*Sqrt2), 0], [-(1/(2*Sqrt2)), 1/(2*Sqrt2), 0], [0, 0, 0.5],
               [-(1/(2*Sqrt2)), -(1/(2*Sqrt2)), 0], [1/(2*Sqrt2), -(1/(2*Sqrt2)), 0], [0, 0, -0.5]])
a1=np.array([[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],
             [[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]]]) #instead of Kronecker product

# Vectors to n.n.n., same sublattice
d = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1],
              [1, -1, 0], [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]])
d45=np.array([[0, 1/Sqrt2, 0], [1/(2*Sqrt2), 1/(2*Sqrt2), 0.5], [-(1/(2*Sqrt2)), 1/(2*Sqrt2), 0.5],
             [0, -(1/Sqrt2),  0], [-(1/(2*Sqrt2)), -(1/(2*Sqrt2)), -0.5], [1/(2*Sqrt2), -(1/(2*Sqrt2)), -0.5],
             [1/Sqrt2, 0, 0], [1/(2*Sqrt2), 1/(2*Sqrt2), -0.5], [-(1/(2*Sqrt2)), 1/(2*Sqrt2), -0.5],
             [-(1/Sqrt2), 0, 0], [-(1/(2*Sqrt2)), -(1/(2*Sqrt2)), 0.5], [1/(2*Sqrt2), -(1/(2*Sqrt2)), 0.5]])
d1=np.array([[[1,1,0],[1,1,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,1,1],[0,1,1]],
             [[1,1,0],[1,1,0],[0,0,0]],[[1,0,1],[0,0,0],[1,0,1]],[[0,0,0],[0,1,1],[0,1,1]],
             [[1,-1,0],[-1,1,0],[0,0,0]],[[1,0,-1],[0,0,0],[-1,0,1]],[[0,0,0],[0,1,-1],[0,-1,1]],
             [[1,-1,0],[-1,1,0],[0,0,0]],[[1,0,-1],[0,0,0],[-1,0,1]],[[0,0,0],[0,1,-1],[0,-1,1]]])


def delta(i , j):
    if i == j:
        return 1
    else:
        return 0


N = int(sys.argv[1]) # number of atoms in first layer for the wire- width
Nh = int(sys.argv[2]) # number of atoms in first layer for the wire- hight
deg = 12  # size of matrix of internal degrees of freedom
nfi = float(sys.argv[3])  # number of flux quanta through the wire
fi = complex(nfi / ((N - 1)*(Nh - 1)))

m = complex(1.65)
t11 = complex(0.5)
t12 = complex(0.9)
t21 = complex(0.9)
Lam1 = complex(0.7)
Lam2 = complex(0.7)

# WIRE HAMILTONIAN WITH MAGNETIC FIELD
def hribb(rx, ry, kz, n):
    hr=np.zeros((12,12),dtype=complex)
    hr += m * delta(rx, 0) * delta(ry, 0) * np.kron(id2, np.kron(tau3, id3))
    for j in range(0, len(a)):
        hr += t12 * delta(rx*1/(2*Sqrt2), a45[j, 0]) * delta(ry*1/(2*Sqrt2), a45[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], a45[j])) * cmath.exp(I*2*Pi*fi*0.5 * (n*ry +0.25*rx*ry)) * np.kron(id2, np.kron((tau1 - I * tau2) * 0.5, a1[j]))
        hr += t21 * delta(rx*1/(2*Sqrt2), a45[j, 0]) * delta(ry*1/(2*Sqrt2), a45[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], a45[j])) * cmath.exp(I*2*Pi*fi*0.5 * (n*ry +0.25*rx*ry)) * np.kron(id2, np.kron((tau1 + I * tau2) * 0.5, a1[j]))
    for j in range(0, len(d)):
        hr += t11 * delta(rx*1/(2*Sqrt2), d45[j, 0]) * delta(ry*1/(2*Sqrt2), d45[j, 1]) * cmath.exp(I * np.dot([0, 0, kz], d45[j])) * cmath.exp(I*2*Pi*fi*0.5 * (n*ry +0.25*rx*ry)) * np.kron(id2, np.kron(tau3, d1[j] * 0.5))
    for k in range(0, 3):
        for j in range(0, 3):
            hr += -I * Lam1 * delta(rx, 0) * delta(ry, 0) * np.kron(
                (np.cross(orb[k], orb[j])[0] * sig[0] + np.cross(orb[k], orb[j])[1] * sig[1] + np.cross(orb[k], orb[j])[2] * sig[2])
                , np.kron((id2 + tau3) * 0.5, orb1[k, j]))
            hr += -I * Lam2 * delta(rx, 0) * delta(ry, 0) * np.kron(
                (np.cross(orb[k], orb[j])[0] * sig[0] + np.cross(orb[k], orb[j])[1] * sig[1] + np.cross(orb[k], orb[j])[2] * sig[2])
                , np.kron((id2 - tau3) * 0.5, orb1[k, j]))
    return hr


def Hwire(kz):
    # Initializing the wire Hamiltonian as a matrix of zeros, size (12*(N*Nh+(N-1)*(Nh-1)) square.
    Hr = np.zeros((deg*( N*Nh + (N-1)*(Nh-1) ), deg*( N*Nh + (N-1)*(Nh-1) )), dtype=complex)
    H = scipy.sparse.lil_matrix(Hr)

    for k in range(0, Nh):  # [0,0] main big blocks
        for i in range(0, N):
            h00 = hribb(0, 0, kz, i)
            for j in range(0, deg):
                for l in range(0, deg):
                    H[k*(deg*N + deg*(N-1))+i*deg+j,k*(deg*N+deg*(N-1))+i*deg+l]=h00[j,l]

    for k in range(0, Nh - 1):
        for i in range(0, N - 1):
            # [0,0] main small blocks
            h00 = hribb(0, 0, kz, i+0.5)
            # [+1,+1] [-1,+1] [+1,-1] [-1,+1] all diagonal blocks
            # hopping from big layer up, and from small layer down
            h11b = hribb(1, 1, kz, i)
            hm1m1b = hribb(-1, -1, kz, i+0.5)
            hm11b = hribb(-1, 1, kz, i+1)
            h1m1b = hribb(1, -1, kz, i+0.5)

            # hopping from small layer up, and from big layer down
            h11s = hribb(1, 1, kz, i+0.5)
            hm1m1s = hribb(-1, -1, kz, i+1)
            hm11s = hribb(-1, 1, kz, i+0.5)
            h1m1s = hribb(1, -1, kz, i)

            for j in range(0, deg):
                for l in range(0, deg):
                    # [0,0] main small blocks
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h00[j,l]
                    # [+1,+1] [-1,+1] [+1,-1] [-1,+1] all diagonal blocks
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, k*(deg*N + deg*(N-1)) + i*deg + l] = h11b[j,l]  # [+1,+1]
                    H[k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = hm1m1b[j,l]  # [-1,-1]
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg + k*(deg*N + deg*(N-1)) + i*deg + l] = hm11b[j,l] # [-1,+1]
                    H[deg + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h1m1b[j,l]  # [+1,-1]

                    H[deg + deg*(N-1) + deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h11s[j,l]  # [+1,+1]
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg + deg*(N-1) + deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = hm1m1s[j,l]  # [-1,-1]
                    H[deg*(N-1) + deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = hm11s[j,l]  # [-1,+1]
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*(N-1) + deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h1m1s[j,l]  # [+1,-1]

    for k in range(0, Nh):  # [2,0],[-2,0] main big blocks
        for i in range(0, N - 1):
            h20 = hribb(2, 0, kz, i)
            hm20 = hribb(-2, 0, kz, i+1)
            for j in range(0, deg):
                for l in range(0, deg):
                    H[k*(deg*N + deg*(N-1)) + (i+1)*deg + j, k*(deg*N + deg*(N-1)) + i*deg + l] = h20[j,l]
                    H[k*(deg*N + deg*(N-1)) + i*deg + j, k*(deg*N + deg*(N-1)) + (i+1)*deg + l] = hm20[j,l]

    for k in range(0, Nh - 1):  # [2,0],[-2,0] main small blocks
        for i in range(0, N - 2):
            h20 = hribb(2, 0, kz, i+0.5)
            hm20 = hribb(-2, 0, kz, i+1.5)
            for j in range(0, deg):
                for l in range(0, deg):
                    H[deg*N + k*(deg*N + deg*(N-1)) + (i+1)*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h20[j, l]
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + (i+1)*deg + l] = hm20[j, l]

    for k in range(0, Nh - 1):  # [0,2],[0,-2] diagonal big blocks
        for i in range(0, N):
            h02 = hribb(0, 2, kz, i)
            h0m2 = hribb(0, -2, kz, i)
            for j in range(0, deg):
                for l in range(0, deg):
                    H[(k+1)*(deg*N + deg*(N-1)) + i*deg + j, k*(deg*N + deg*(N-1)) + i*deg + l] = h02[j,l]
                    H[k*(deg*N + deg*(N-1)) + i*deg + j, (k+1)*(deg*N + deg*(N-1)) + i*deg + l] = h0m2[j,l]

    for k in range(0, Nh - 2):  # [0,2],[0,-2] diagonal small blocks
        for i in range(0, N - 1):
            h02 = hribb(0, 2, kz, i+0.5)
            h0m2 = hribb(0, -2, kz, i+0.5)
            for j in range(0, deg):
                for l in range(0, deg):
                    H[deg*N + (k+1)*(deg*N + deg*(N-1)) + i*deg + j, deg*N + k*(deg*N + deg*(N-1)) + i*deg + l] = h02[j,l]
                    H[deg*N + k*(deg*N + deg*(N-1)) + i*deg + j, deg*N + (k+1)*(deg*N + deg*(N-1)) + i*deg + l] = h0m2[j, l]

    return H

# Creating k values for wire band structure calculations
ki = float(sys.argv[4])*Pi  # left limit of the spectrum
kf = float(sys.argv[5])*Pi  # right limit of the spectrum
kw_points = 51  # number of k points per calculation
divide_k = int(sys.argv[6])  # the value in which to increase density of k points
divide_k_index = int(sys.argv[7])  # numbering the sub calculations to get complex band structure
kw_val = np.linspace(ki, kf, kw_points)


def offset(index):
    return np.float64((kf-ki)*(index-1)/((kw_points-1)*divide_k))


# Creating the wire band structure data: an array of the eigenvalues at all k-points
num_ev = 120
wbands = np.zeros((kw_points, num_ev), dtype=float)

for i in range(0, kw_points):
    h = Hwire(kw_val[i]+offset(divide_k_index))
    w, v = scipy.sparse.linalg.eigsh(h, num_ev, sigma=-0.075)
    wbands[i] = w

file_name='Wire110_'+str(N)+'W_'+str(Nh)+'H_'+str(num_ev)+'ev_'+str(kw_points)+'kp_'+str(divide_k_index)+'_of_'+str(divide_k)+'_'+str(nfi)+'fi'

save(file_name+'.npy', wbands)



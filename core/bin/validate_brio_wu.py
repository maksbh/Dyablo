import pyablo
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import sys

rasterize = (sys.argv[-1] == '--rasterize')

athena_fn = "brio_wu_reference.npy"
dyablo_fn = "test_brio_wu_main.xmf"

## Reading the Athena++ run
if not os.path.exists(athena_fn):
    print(f"ERROR : File '{athena_fn}' not found in current folder")
    exit(1)

athena_dat = np.load(athena_fn)

## Reading dyablo run
if not os.path.exists(dyablo_fn):
    print(f"ERROR : File '{dyablo_fn}' not found in current folder")
    exit(1)

gamma0 = 2.0
    
reader = pyablo.XdmfReader()
time_series = reader.readTimeSeries(dyablo_fn)
snap = reader.readSnapshot(time_series[-1])

Nx = 1024
dx = 1.0/Nx
x = np.linspace(0.5*dx, 1.0-0.5*dx, Nx)
y = np.ones_like(x) * 0.5
z = np.zeros_like(x)

# Ugly
pos = [(x[i], y[i], z[i]) for i in range(Nx)]
cids = list(sorted(set(snap.getCellsFromPositions(pos))))

pos = np.array(snap.getCellsCenter(cids))
x = pos[:,0]

read_array = lambda arr: np.array(snap.readAllFloat(arr))[cids]

rho = read_array('rho')
By  = read_array('By')
Bx  = read_array('Bx')
rho_u = read_array('rho_vx')
rho_v = read_array('rho_vy')
e_tot = read_array('e_tot') 

Ek = 0.5 * (rho_u**2.0 + rho_v**2.0) / rho
Em = 0.5 * (Bx**2.0 + By**2.0)
p = (e_tot - Ek - Em) * (gamma0-1.0)

fig, ax = plt.subplots(1, 3, figsize=(12, 3), layout='constrained')
ax[0].plot(x, rho, '.r', label='Dyablo', rasterized=rasterize)
ax[0].plot(athena_dat[:,0], athena_dat[:,1], '-k', label='Athena (reference)', rasterized=rasterize)
ax[1].plot(x, p, '.r', rasterized=rasterize)
ax[1].plot(athena_dat[:,0], athena_dat[:,2], '-k', rasterized=rasterize)
ax[2].plot(x, By, '.r', rasterized=rasterize)
ax[2].plot(athena_dat[:,0], athena_dat[:,3], '-k', rasterized=rasterize)

ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$\rho$')
ax[1].set_xlabel('x')
ax[1].set_ylabel('$p$')
ax[2].set_xlabel('x')
ax[2].set_ylabel('$B_y$')
plt.savefig('brio_wu.pdf', dpi=150)
plt.show()

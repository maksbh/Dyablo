import pyablo
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import newton
from scipy import integrate
import scipy.constants as cst
import h5py

xmf_filename = sys.argv[1] #"test_zeldovitch_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]

print("Validate Zeldovitch")
print(f'XMF filename : {xmf_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')

# COSMO FUNCTIONS
def ddplus(a, om, ov):
    if (a==0.0):
        return 0
        
    eta = np.sqrt(om/a+ov*a*a+1.0-om-ov)
    return 2.5/(eta*eta*eta)

#=======================================================

def dplus(a,omegam,omegav):
    eta = np.sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav)
    return eta/a*integrate.romberg(lambda a : ddplus(a, omegam, omegav), 0., a)

#=======================================================

def dladt(a,omegam,omegav):
    eta = np.sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav)
    return a*eta

#=======================================================

def fomega(a,omegam,omegav) :                                                    
    if (omegam==1.0 and omegav==0.0):
        return 1.0
    
    omegak = 1.0-omegam-omegav
    eta = np.sqrt(omegam/a+omegav*a*a+omegak)
    return (2.5/dplus(a,omegam,omegav)-1.5*omegam/a-omegak)/(eta*eta)

class zeldovitch_analytical:
  def __init__(self, across, asnap, omegam, omegav, omegab, G, H0, Lbox ):
    
    rhoc = 3 * H0**2 / (8 * np.pi * G)
    rhostar = omegam * rhoc
    tstar = 2.0 / H0 / np.sqrt(omegam)
    rstar = Lbox 
    vstar = rstar / tstar

    self.rho_fact = omegab * rhoc / rhostar
    self.dplus_ratio = dplus(asnap ,omegam,omegav) / dplus(across ,omegam,omegav)
    self.vfact = (self.dplus_ratio * Lbox / (2*np.pi) * fomega(asnap ,omegam,omegav)*dladt(asnap ,omegam,omegav) * H0) / vstar

    self.qgrid = np.linspace(0,1,100000)
    self.xgrid = (self.qgrid + self.dplus_ratio/(2*np.pi) * Lbox * np.sin(2*np.pi*self.qgrid)/rstar)

  def value( self, x ): 
    q = np.interp(x, self.xgrid, self.qgrid)

    rho = self.rho_fact/( 1 + self.dplus_ratio * np.cos(2*np.pi*q) )
    vx = self.vfact * np.sin(2*np.pi*q)
    return (rho,vx)
    
    
# Physics
omegam=0.3
omegav=0.7
omegab=0.049
# G = cst.G
# H0 = 2.1753246753246754e-18 
# Lbox = 3.08e23
G = 1
H0 = 1
Lbox = 1
across=0.16666666 #crossing expansion factor
asnap=0.16666666 #crossing expansion factor

# Open output file
reader = pyablo.XdmfReader()
series = reader.readTimeSeries(xmf_filename)
snap = reader.readSnapshot(series[-1])

fields_xmf_filename = series[-1]
hdf5_filename = fields_xmf_filename[0:-3]+"h5"
particles_h5_filename = hdf5_filename[0:10] + "_particles_particles" + hdf5_filename[10:]

NCells = snap.getNCells()
ids = np.arange( 0, NCells )

# Load fields
px = np.array(snap.getCellsCenter( ids ))[:,0]
rho = np.asarray( snap.readAllFloat('rho') )
rho_u = np.asarray( snap.readAllFloat('rho_vx') )

## Sorting is only needed for plot
sort_order = np.argsort( px )
px = px[sort_order]
rho = rho[sort_order]
rho_u = rho_u[sort_order]

u = rho_u/rho

fpart = h5py.File(particles_h5_filename, 'r')

particle_px = np.array(fpart['coordinates'])[:,0]
particle_vx = np.array(fpart['vx'])

sort_order_part = np.argsort( particle_px )
particle_px = particle_px[sort_order_part]
particle_vx = particle_vx[sort_order_part]

formula = zeldovitch_analytical(across, asnap, omegam, omegav, omegab, G, H0, Lbox)

analytical = np.asarray([ formula.value( x ) for x in px ])

expected_rho = analytical[:,0]
expected_u = analytical[:,1]

analytical_part = np.asarray([ formula.value( x ) for x in particle_px ])
expected_upart = np.asarray(analytical_part[:,1])

# Computing L1 norm
L1_rho   = np.linalg.norm(rho - expected_rho, ord=1)   / NCells
L1_u   = np.linalg.norm(u - expected_u, ord=1)   / NCells
L1_upart = np.linalg.norm(particle_vx - expected_upart, ord=1) / particle_vx.size

print( f'Cells Velocity; L1 error = {L1_u:.4}' )
print( f'Particles Velocity; L1 error = {L1_upart:.4}' )

fig, ax = plt.subplots(2, 1, figsize=(7, 7))
ax[0].plot(px, expected_rho, '--', label = "Expected")
ax[0].plot(px, rho, '+', label = "Cells")
ax[0].set_title(f'Density; L1 error = {L1_rho:.4}')
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$\rho$')
ax[1].plot(particle_px, expected_upart, '--', label = "Expected")
ax[1].plot(px, u, '+', label = f'Cells (L1 error = {L1_u:.4})')
ax[1].plot(particle_px, particle_vx, '+', label = f'Particles (L1 error = {L1_upart:.4})')
ax[1].set_title(f'Velocity')
ax[1].set_xlabel('x')
ax[1].set_ylabel(r'$v_x$')
plt.legend()
plt.suptitle(f'Zeldovitch')
plt.legend()
plt.tight_layout()
plt.savefig(f'{png_filename}')
print( f'Exported {png_filename}' )

if( L1_u > target_precision ):
  print( f'Precision target not met {L1_u:.4} > {target_precision:.4}'  )
  exit(1)

if( L1_upart > target_precision ):
  print( f'Precision target not met {L1_upart:.4} > {target_precision:.4}'  )
  exit(1)
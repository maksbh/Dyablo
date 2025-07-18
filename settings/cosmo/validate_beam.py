import pyablo
import numpy as np
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from scipy.signal import argrelmin


### Read ini file and build configuration
def read_config(filename) :
    config = ConfigParser(inline_comment_prefixes=('#',';'))
    config.read(filename)

    # Calculating domain size from the data
    bx = config.getint('amr', 'bx')
    by = config.getint('amr', 'by')
    bz = config.getint('amr', 'bz')

    lmin = config.getint('amr', 'level_min')
    lmax = config.getint('amr', 'level_max')

    cor_x = config.getint('amr', 'coarse_oct_resolution_x')
    cor_y = config.getint('amr', 'coarse_oct_resolution_y')
    cor_z = config.getint('amr', 'coarse_oct_resolution_z')

    xmin = config.getfloat('mesh', 'xmin')
    xmax = config.getfloat('mesh', 'xmax')
    ymin = config.getfloat('mesh', 'ymin')
    ymax = config.getfloat('mesh', 'ymax')
    zmin = config.getfloat('mesh', 'zmin')
    zmax = config.getfloat('mesh', 'zmax')

    # Number of cells
    Nx = bx*cor_x
    Ny = by*cor_y
    Nz = bz*cor_z

    dz = (zmax-zmin)/Nz
    
    source_position = config.getfloat('rad', 'source_position')
    spawn_rate = config.getfloat('rad', 'spawn_rate')
    cfrac = config.getfloat('cosmology', 'clight_fraction')


    conf = {"bx":bx, "by":by, "bz":bz, "lmin":lmin, "lmax":lmax, "cor_x":cor_x, "cor_y":cor_y, 
            "cor_z":cor_z, "xmin":xmin, "xmax":xmax, "ymin":ymin, "ymax":ymax, "zmin":zmin, "zmax":zmax, "Nx":Nx, "Ny":Ny, "Nz":Nz, "dz":dz,
            "source_position":source_position, "spawn_rate":spawn_rate, "cfrac":cfrac}

    return conf



############################################################################################

def compute_diff(files, conf) :

  # Middle position in pixels
  pos = (int)(conf["Nx"]/2)

  # x position of the source in Mpc
  x0 = conf["source_position"]/5

  # speed of light fraction
  cfrac = conf["cfrac"]

  mpc = 3.085678e22 #m
  c = 299792458*cfrac #m/s
  myr = 3600*24*365*1e6 #s
  mol = 6.02214076e23 

  times = []
  ratio_distances = []
  ratio_nb_photons = []

  for f in files:
  
      dx3 = conf['xmax'] * conf['xmax'] * conf['xmax'] /conf['Nx']/conf['Nx']/conf['Nx'] 
      
      # Read snapshot
      snap = reader.readSnapshot(f) 
      mask = snap.getSortingMask3d(conf["lmin"], conf["bx"], conf["by"], conf["bz"], conf["cor_x"], conf["cor_y"], conf["cor_z"])

      # Get all e_rad values
      erad = np.array(snap.readAllFloat('e_rad'))[mask].reshape((conf["Nz"], conf["Ny"], conf["Nx"]))
      
      # Sum of erad values in a single line that goes through the middle of the beam
      sum = np.sum(erad[pos], axis=0)
      
      # Time in Myr
      t = snap.getTime()
      
      nb_photons_theo = conf["spawn_rate"] * t
      nb_photons = np.sum(erad)*dx3

      # Distance = x0 + (vitesse de la lumière x temps de simu)
      distance_theo = x0 + t*c*myr/mpc 

      # Compute derivative
      x = np.linspace(conf["xmin"], conf["xmax"], conf["Nz"])
      dx = np.diff(x,1)
      dy = np.diff(sum,1)
      dv = np.array(dy/dx)
      x = 0.5*(x[:-1]+x[1:])

      # Get local minima from the derivative and remove noisy points at the end
      peaks = argrelmin(dv, order=5)[0]
      if abs(dv[peaks[-1]]) < 1e-10 :
          peaks = peaks[:-1]
      
      index = -1
      distance = x[peaks[index]+1]
      
      ratio_distances.append((distance-distance_theo)/distance_theo)
      ratio_nb_photons.append((nb_photons-nb_photons_theo)/nb_photons_theo)
      times.append(t)

      print('t=', t, ' distance_theo=', round(distance_theo,2), ' distance=', round(distance,2), ' ratio_distance=', round((distance-distance_theo)/distance_theo,2), 
            'nb_photons_theo=', round(nb_photons_theo,2), ' nb_photons=', round(nb_photons,2), ' ratio_photons=', round((nb_photons-nb_photons_theo)/nb_photons_theo,4) )
    
  return times, ratio_distances, ratio_nb_photons


############################################################################################

  
xmf_filename = sys.argv[1] #"test_beam_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]
if( len(sys.argv) >= 5 ):
  ini_filename = sys.argv[4]
else:
  ini_filename="last.ini"

print("Validate photons beam")
print(f'XMF filename : {xmf_filename}')
print(f'ini filename : {ini_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')  

# Open output file
reader = pyablo.XdmfReader()

# List of output files
series = reader.readTimeSeries(xmf_filename)
files = series[1:]

conf = read_config(ini_filename)

times, ratio_distances, ratio_nb_photons = compute_diff(files, conf)

last_ratio_distance = abs(ratio_distances[-1])
last_ratio_nb_photons = abs(ratio_nb_photons[-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.plot(times, ratio_distances)
ax1.set_xlabel('Time [Myr]')
ax1.set_ylabel('(distance - distance_theo)/distance_theo')
ax1.set_title(f'L1 error = {last_ratio_distance:.4}')
ax1.grid('.')

ax2.plot(times, ratio_nb_photons)
ax2.set_xlabel('Time [Myr]')
ax2.set_ylabel('(nb_photons-nb_photons_theo)/nb_photons_theo')
ax2.set_title(f'L1 error = {last_ratio_nb_photons:.4}')
ax2.grid()

plt.suptitle('Photons beam')
plt.savefig(f'{png_filename}')
fig.tight_layout()


print( f'Exported {png_filename}' )
print( f'Difference distance = {last_ratio_distance:.4}'  )
print( f'Difference photons = {last_ratio_nb_photons:.4}'  )

if( last_ratio_distance > target_precision ):
  print( f'Precision target not met for front distance {last_ratio_distance:.4} > {target_precision:.4}'  )
  exit(1)
  
if( last_ratio_nb_photons > target_precision ):
  print( f'Precision target not met for photons conservation {last_ratio_nb_photons:.4} > {target_precision:.4}'  )
  exit(1)
  
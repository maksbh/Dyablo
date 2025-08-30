import pyablo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from configparser import ConfigParser
from scipy.signal import argrelmin

matplotlib.use("Agg")

class Results :
    def __init__(self) :
        self.times = []
        self.distance = []
        self.distance_theo = []
        self.nb_photons = []
        self.nb_photons_theo = []
        self.ratio_distances = []
        self.ratio_nb_photons = []
        
    def append(self, time, distance, distance_theo, nb_photons, nb_photons_theo) :
        self.times.append(time)
        self.distance.append(distance)
        self.distance_theo.append(distance_theo)
        self.nb_photons.append(nb_photons)
        self.nb_photons_theo.append(nb_photons_theo)
        self.ratio_distances.append((distance-distance_theo)/distance_theo)
        self.ratio_nb_photons.append((nb_photons-nb_photons_theo)/nb_photons_theo)
        
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
    ctilde = config.getfloat('cosmology', 'ctilde')

    conf = {"bx":bx, "by":by, "bz":bz, "lmin":lmin, "lmax":lmax, "cor_x":cor_x, "cor_y":cor_y, 
            "cor_z":cor_z, "xmin":xmin, "xmax":xmax, "ymin":ymin, "ymax":ymax, "zmin":zmin, "zmax":zmax, "Nx":Nx, "Ny":Ny, "Nz":Nz, "dz":dz,
            "source_position":source_position, "spawn_rate":spawn_rate, "ctilde":ctilde}

    return conf



############################################################################################

def compute_diff(files, conf) :

  # Middle position in pixels
  pos = (int)(conf["Nx"]/2)

  # x position of the source in Mpc
  x0 = conf["source_position"]/5

  # speed of light fraction
  ctilde = conf["ctilde"]

  results = Results()

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
      distance_theo = x0 + t*ctilde 

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
      
      results.append(t, distance, distance_theo, nb_photons, nb_photons_theo)    

      print('t=', t, ' distance_theo=', round(distance_theo,2), ' distance=', round(distance,2), ' ratio_distance=', round((distance-distance_theo)/distance_theo,2), 
            'nb_photons_theo=', round(nb_photons_theo,2), ' nb_photons=', round(nb_photons,2), ' ratio_photons=', round((nb_photons-nb_photons_theo)/nb_photons_theo,4) )
    
  return results


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

results = compute_diff(files, conf)

last_ratio_distance = abs(results.ratio_distances[-1])
last_ratio_nb_photons = abs(results.ratio_nb_photons[-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.plot(results.times, results.distance, label="distance")
ax1.plot(results.times, results.distance_theo, label="distance_theo")
ax1.set_xlabel('Time [Myr]')
ax1.set_ylabel('(distance [Mpc]')
ax1.set_title(f'L1 error = {last_ratio_distance:.4}')
ax1.grid()
ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

ax2.plot(results.times, results.nb_photons, label="nb_photons")
ax2.plot(results.times, results.nb_photons_theo, label="nb_photons_theo")
ax2.set_xlabel('Time [Myr]')
ax2.set_ylabel('Nb photons')
ax2.set_title(f'L1 error = {last_ratio_nb_photons:.4}')
ax2.grid()
ax2.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

fig.tight_layout()
plt.savefig(f'{png_filename}', dpi=100)

print( f'Exported {png_filename}' )
print( f'Difference distance = {last_ratio_distance:.4}'  )
print( f'Difference photons = {last_ratio_nb_photons:.4}'  )

if( last_ratio_distance > target_precision ):
  print( f'Precision target not met for front distance {last_ratio_distance:.4} > {target_precision:.4}'  )
  exit(1)
  
if( last_ratio_nb_photons > target_precision ):
  print( f'Precision target not met for photons conservation {last_ratio_nb_photons:.4} > {target_precision:.4}'  )
  exit(1)

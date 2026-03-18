import numpy as np
import matplotlib.pyplot as plt
import pyablo
import os
import sys

def plot_profiles(folder, mhd_run=False, n=5, debug_mhd=False):
    cwd = os.getcwd()
    try:
        os.chdir(folder)
    except:
        print('Unknown folder :', folder)

    reader = pyablo.XdmfReader()

    series = reader.readTimeSeries('heat_conduction_main.xmf')

    gamma0 = 5.0/3.0

    lpos = [(0.75, 0.2), (0.45, 0.2), (0.22, 0.2), (0.02, 0.2), (0.0, 0.6)]
    
    colors = ['black', 'green', 'slateblue', 'darkorange', 'dodgerblue']
    i=0
    fig, ax = plt.subplots(1, 1)
    for snap_name in series[:n]:
        try:
            snap = reader.readSnapshot(snap_name)
        except:
            print(f'Error loading {snap_name}')
            os.chdir('..')
            return
        N = snap.getNCells()
        cid = list(range(N))
        x = np.array(snap.getCellsCenter(cid))[:,0]
        x = np.array(sorted(list(set(list(x)))))
        N = x.shape[0]+2
        xfull = np.empty(N)
        xfull[1:-1] = x
        xfull[0] = 0.0
        xfull[-1] = 1.0
        y = np.array([0.001]*(N-2))
        z = np.array([0.0]*(N-2))
        pos = np.stack((x, y, z)).T
        
        rho   = np.array(snap.probeQuantity(pos, 'rho'))
        rho_u = np.array(snap.probeQuantity(pos, 'rho_vx'))
        e_tot = np.array(snap.probeQuantity(pos, 'e_tot'))
        Ek = 0.5 * rho_u**2.0/rho
        Em = 0.0
        if mhd_run:
            Bx = np.array(snap.probeQuantity(pos, 'Bx'))
            Em = 0.5 * Bx*Bx
        T  = (e_tot - Ek - Em) * (gamma0-1.0) / rho
    
        Tfull = np.empty(N)
        Tfull[1:-1] = T
        Tfull[0] = 0.1
        Tfull[-1] = 1.0
        t = snap.getTime()
        ax.plot(xfull, Tfull, color=colors[i])
        ax.text(lpos[i][0], lpos[i][1], f't={t:.2f}', color=colors[i]) 
        ax.set_xlabel('x')
        ax.set_ylabel('Temperature')
        
        if snap_name == series[n-1]:
            T_ = T
            x_ = x
        i+=1

    analytical_res = lambda x: (0.1**3.5 + (1.0-0.1**3.5)*x)**(2.0/7.0)
    asymp_T = analytical_res(xfull)
    asymp_L1 = analytical_res(x_)
    
    L1_norm = np.linalg.norm(asymp_L1 - T_, ord=1) / len(T_)
    
    ax.plot(xfull, asymp_T, linestyle='--', color='black', label='Asymptotic solution')
    plt.suptitle(f'Heat conduction test, L1_norm={L1_norm}')
    os.chdir(cwd)
    filename_out = sys.argv[3]
    plt.savefig(filename_out)
    print(f'Generated test figure in {filename_out}')
    return L1_norm

if len(sys.argv) < 4:
    print(f'USAGE: python3 {sys.argv[0]} PATH PRECISION OUTPUT_FILENAME')
    exit(1)
    
L1_target = float(sys.argv[2])
L1_norm = plot_profiles(sys.argv[1])

if L1_norm > L1_target:
    print(f'Precision target not met {L1_norm:.4} > {L1_target:.4}')
    exit(1)

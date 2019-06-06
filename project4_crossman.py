# -*- coding: utf-8 -*-
"""
@Title:     Physics 660 Project Four
@Author     Andrew Crossman
@Date       April 20th, 2019
"""
###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
__version__ = '1.0'
print(__doc__)
print('version:', __version__)
###############################################################################
# FUNCTION DEFINITIONS
###############################################################################
def total_energy(x, x_old, n_particles, dt):
    """total energy of the system"""
    v = (x-x_old)/dt
    dx = (x + x_old - np.roll(x, 1) - np.roll(x_old, 1)) / 2.0
    dx[0] = 0
    v[0] = 0
    return np.sum(0.5*v**2 + 0.5*dx**2)

def velocities(x, x_old, dt):
    v = (x-x_old)/dt
    return v

def mode_energy(x, x_old, n_particles, k, dt):
    """energy of mode k"""
    tmp = np.arange(0, n_particles + 2, 1)
    tmp[-1] = 0
    m = np.sqrt(2.0 / (n_particles + 1)) * np.dot(x, np.sin((tmp*k*np.pi/(n_particles + 1))))
    m_old = np.sqrt(2.0 / (n_particles + 1)) * np.dot(x_old, np.sin((tmp*k*np.pi/(n_particles + 1))))
    return 0.5 * (m-m_old)**2 / dt**2 + 0.5 * (m+m_old)**2 * np.sin(np.pi*k/(2*(n_particles + 1)))**2

def mode_energies(x, x_old, n_particles, dt):
    """"energy across all k modes"""
    energies = []
    for k in list(range(1,n_particles+1)):
        tmp = np.arange(0, n_particles + 2, 1)
        tmp[-1] = 0
        m = np.sqrt(2.0 / (n_particles + 1)) * np.dot(x, np.sin((tmp*k*np.pi/(n_particles + 1))))
        m_old = np.sqrt(2.0 / (n_particles + 1)) * np.dot(x_old, np.sin((tmp*k*np.pi/(n_particles + 1))))
        energies.append(0.5 * (m-m_old)**2 / dt**2 + 0.5 * (m+m_old)**2 * np.sin(np.pi*k/(2*(n_particles + 1)))**2)
    return(energies)

def mode_amplitude(x, n_particles, k, amplitude):
    """"eignevector coefficient of mode k"""
    #for n in list(range(0,N_mass+2)): #normal mode
    #print(x)
    An = 0
    for m in list(range(0,n_particles+2)): #mass number
        An = An + x[m]*np.sin(k*np.pi*m/(n_particles+1))
    An = amplitude*An
    return(An)
    
def force(x):
    """Force on (n_particle +  2)/2"""
    p = int(x.size / 2)
    f = x[p + 1] + x[p - 1] - 2 * x[p] 
    return f

def QuestionOne(B,mode):
    # INITIALIZING VARIABLES
    ## number of particles (+2 for the stationary ones at the edges)
    n_particles = 32
    ## timestep dt
    dt = 0.05
    ## maximal time
    t_max = 15000
    w = 2*np.absolute(np.sin(np.pi*mode/(2*(n_particles+1))))
    t_max = 3/2*int(400*np.pi/(w))
    # Set beta
    beta = B
    # SETTING STARTPOSTION
    #change the initial mode to get different harmonics
    init_mode = mode
    amplitude = np.sqrt(2/(n_particles+1))
    x = x_old = [0.0 for i in list(range(0,n_particles+2))]
    tmp = np.arange(0, n_particles+2, 1)
    x = x_old = 10*amplitude*np.sin(init_mode*np.pi*tmp/(n_particles+1))
    x[0] = x[-1] = x_old[0] = x_old[-1] = 0
    
    pos = np.linspace(0, n_particles + 2, num=n_particles + 2)
    plt.title('Start position of particles')
    plt.plot(pos, x, 'ro')
    plt.show()

    # CALCULATING x, total energy and mode energy   
    print('Calculating...')   
    t = 0
    data = {                                                                \
            'x':[x],                                                        \
            'v':[velocities(x, x_old, t)],                                                        \
            'total_energy':[total_energy(x, x_old, n_particles, dt)],       \
            'mode1':[mode_energy(x, x_old, n_particles, 1, dt)],            \
            'mode2':[mode_energy(x, x_old, n_particles, 2, dt)],            \
            'mode3':[mode_energy(x, x_old, n_particles, 3, dt)],            \
            'mode4':[mode_energy(x, x_old, n_particles, 4, dt)],            \
            'mode5':[mode_energy(x, x_old, n_particles, 5, dt)],            \
            'mode6':[mode_energy(x, x_old, n_particles, 6, dt)],            \
            'mode7':[mode_energy(x, x_old, n_particles, 7, dt)],            \
            'mode8':[mode_energy(x, x_old, n_particles, 8, dt)],            \
            'amp1':[mode_amplitude(x, n_particles, 1, amplitude)],            \
            'amp2':[mode_amplitude(x, n_particles, 2, amplitude)],            \
            'amp3':[mode_amplitude(x, n_particles, 3, amplitude)],            \
            'amp4':[mode_amplitude(x, n_particles, 4, amplitude)],            \
            'amp5':[mode_amplitude(x, n_particles, 5, amplitude)],            \
            'amp6':[mode_amplitude(x, n_particles, 6, amplitude)],            \
            'amp7':[mode_amplitude(x, n_particles, 7, amplitude)],            \
            'amp8':[mode_amplitude(x, n_particles, 8, amplitude)],          \
            'force':[force(x)]
            }
    dtq = dt**2
    while t < t_max:
        #calculating new position of particles
        x, x_old = (np.roll(x,-1) + np.roll(x,1) - 2.0*x) * dtq                   \
                    + beta*((np.roll(x,-1)-x)**3 - (x-np.roll(x,1))**3 ) * dtq     \
                    + 2.0 * x - x_old,  x 
        x[0] = x[-1] = 0
        
        data['x'].append(x)
        data['v'].append(velocities(x, x_old, dt))
        data['total_energy'].append(total_energy(x, x_old, n_particles, dt))
        data['mode1'].append(mode_energy(x, x_old, n_particles, 1, dt))
        data['mode2'].append(mode_energy(x, x_old, n_particles, 2, dt))
        data['mode3'].append(mode_energy(x, x_old, n_particles, 3, dt))
        data['mode4'].append(mode_energy(x, x_old, n_particles, 4, dt))
        data['mode5'].append(mode_energy(x, x_old, n_particles, 5, dt))
        data['mode6'].append(mode_energy(x, x_old, n_particles, 6, dt))            
        data['mode7'].append(mode_energy(x, x_old, n_particles, 7, dt))  
        data['mode8'].append(mode_energy(x, x_old, n_particles, 8, dt))
        data['amp1'].append(mode_amplitude(x, n_particles, 1, amplitude))  
        data['amp2'].append(mode_amplitude(x, n_particles, 2, amplitude))
        data['amp3'].append(mode_amplitude(x, n_particles, 3, amplitude)) 
        data['amp4'].append(mode_amplitude(x, n_particles, 4, amplitude)) 
        data['amp5'].append(mode_amplitude(x, n_particles, 5, amplitude)) 
        data['amp6'].append(mode_amplitude(x, n_particles, 6, amplitude)) 
        data['amp7'].append(mode_amplitude(x, n_particles, 7, amplitude))                    
        data['amp8'].append(mode_amplitude(x, n_particles, 8, amplitude))
        data['force'].append(force(x))
        t += dt
    data['time'] = np.linspace(0, len(data['x']), num=len(data['x'])) * dt
    #convert to numpy-array:
    for key in data.keys():
        data[key] = np.array(data[key])
    print('done')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creates Plot for Energies vs Time
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(data['time'], data['mode1'], label='Mode 1')
    ax.plot(data['time'], data['mode2'], label='Mode 2')
    ax.plot(data['time'], data['mode3'], label='Mode 3')
    ax.plot(data['time'], data['mode4'], label='Mode 4')
    ax.plot(data['time'], data['mode5'], label='Mode 5')
    ax.plot(data['time'], data['mode6'], label='Mode 6')
    ax.plot(data['time'], data['mode7'], label='Mode 7')
    ax.plot(data['time'], data['mode8'], label='Mode 8')
    ax.set_xlabel('Steps',style='italic',fontsize=14)
    ax.set_ylabel('Energy',style='italic',fontsize=14)
    ax.set_title('Mode energy',style='italic',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("EnergiesN="+str(mode)+"B="+str(beta)+".png", bbox_inches='tight',dpi=600)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creates PLot for Eignvector Coefficients vs Time
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(data['time'], data['amp1'], label='Mode 1')
    ax1.plot(data['time'], data['amp2'], label='Mode 2')
    ax1.plot(data['time'], data['amp3'], label='Mode 3')
    ax1.plot(data['time'], data['amp4'], label='Mode 4')
    ax1.plot(data['time'], data['amp5'], label='Mode 5')
    ax1.plot(data['time'], data['amp6'], label='Mode 6')
    ax1.plot(data['time'], data['amp7'], label='Mode 7')
    ax1.plot(data['time'], data['amp8'], label='Mode 8')
    ax1.set_xlabel('Steps',style='italic',fontsize=14)
    ax1.set_ylabel('Eigenvector Coefficient',style='italic',fontsize=14)
    ax1.set_title('Mode Amplitudes',style='italic',fontsize=16)
    ax1.legend(loc="upper right")
    f1.show()
    f1.savefig("AmplitudesN="+str(mode)+"B="+str(beta)+".png", bbox_inches='tight',dpi=600)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creates Poincare sections
    px, pv = [], []
    count = 0
    for i in data['amp3']:
        if i<10**-1 and i>-10**-1:
            px.append(data['amp1'][count])
            pv.append(mode_amplitude(data['v'][count], n_particles, 1, amplitude))
        count+=1
    f2,ax2 = plt.subplots()
    f2.tight_layout()
    ax2.scatter(px,pv, c='k')
    ax2.set_xlabel(r'$D_{\ell=1}$' ,style='italic',fontsize=14)
    ax2.set_ylabel(r'$\frac{d}{dt}D_{\ell=1}$',style='italic',fontsize=14)
    ax2.set_title('FPU System N=32 '+r'$\beta=$'+str(beta)+' '+r'$\ell=1$' ,style='italic',fontsize=16)
    f2.show()
    f2.savefig("Poincare5aN="+str(mode)+"B="+str(beta)+".png", bbox_inches='tight', dpi=600)
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def QuestionTwo(B,mod):
    n_particles = 32
    dt = 0.05
    w = 2*np.absolute(np.sin(np.pi*mod/(2*(n_particles+1))))
    t_max = 3/2*int(400*np.pi/(w))
    #t_max = 5000
    beta = B
    #SET STARTING POSTIONs
    init_mode = mod
    amplitude = np.sqrt(2/(n_particles+1))
    x = x_old = [0.0 for i in list(range(0,n_particles+2))]
    tmp = np.arange(0, n_particles+2, 1)
    x = x_old = 10*amplitude*np.sin(init_mode*np.pi*tmp/(n_particles+1))
    x[0] = x[-1] = x_old[0] = x_old[-1] = 0
    #PLOT INITIAL POSITIONS
    pos = np.linspace(0, n_particles + 2, num=n_particles + 2)
    plt.title('Start position of particles')
    plt.plot(pos, x, 'ro')
    plt.show()
    #INITAITE DICTIONARY 
    print('Calculating...')   
    t = 0
    data = {                                                                \
            'x':[x],                                                        \
            'total_energy':[total_energy(x, x_old, n_particles, dt)],       \
            'amps':[mode_energies(x, x_old, n_particles, dt)]
            }
    dtq = dt**2
    #LOOP THROUGH ALL TIMES ~300 PERIODS
    while t < t_max:
        #calculating new position of particles
        x, x_old = (np.roll(x,-1) + np.roll(x,1) - 2.0*x) * dtq                   \
                    + beta*((np.roll(x,-1)-x)**3 - (x-np.roll(x,1))**3 ) * dtq     \
                    + 2.0 * x - x_old,  x 
        x[0] = x[-1] = 0
        data['x'].append(x)
        data['total_energy'].append(total_energy(x, x_old, n_particles, dt)) 
        data['amps'].append(mode_energies(x, x_old, n_particles, dt))  
        t += dt
    data['time'] = np.linspace(0, len(data['x']), num=len(data['x'])) * dt
    #convert to numpy-array:
    for key in data.keys():
        data[key] = np.array(data[key])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Calculates sigma_e(t)
    sigma_e = []
    #find the average value for each E_l
    avgs = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0, \
           '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0}
    for t in list(range(0,len(data['time']))):
        for k in list(range(0,n_particles)):
            avgs[str(k+1)]+=data['amps'][t][k]
    for mode in avgs.keys():
        avgs[mode]=avgs[mode]/len(data['x'])
    #find each value of simga_e(t)
    print(len(data['time']))
    for t in list(range(0,len(data['time']))):
        value = 0
        for k in list(range(0,n_particles)):
            value += (data['amps'][t][k] - avgs[str(k+1)])**2
        sigma_e.append(np.sqrt(value/(n_particles-1)))
    #find e
    e = data['amps'][0][mod-1]/(n_particles)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Creates Plot for Energies vs Time
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(data['time'], sigma_e, 'k')
    ax.set_xlabel('Steps',fontsize=14)
    ax.set_ylabel(r'$\sigma_E$',fontsize=14)
    ax.set_title(r'$\sigma_E vs Time$',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("sigmaMode="+str(mod)+"B="+str(beta)+".png", bbox_inches='tight',dpi=600)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def QuestionThree(mod):
    n_particles = 32
    dt=.05
    betas = []
    e_crits = []
    amplitudes = [2,3,4,5,6,7,8,8.5,9,9.5,10]
    for a in amplitudes:
        #SET STARTING POSTIONs
        init_mode = mod
        amplitude = np.sqrt(2/(n_particles+1))
        x = x_old = [0.0 for i in list(range(0,n_particles+2))]
        tmp = np.arange(0, n_particles+2, 1)
        x = x_old = a*amplitude*np.sin(init_mode*np.pi*tmp/(n_particles+1))
        x[0] = x[-1] = x_old[0] = x_old[-1] = 0
        #PLOT INITIAL POSITIONS
        '''
        pos = np.linspace(0, n_particles + 2, num=n_particles + 2)
        plt.title('Start position of particles')
        plt.plot(pos, x, 'ro')
        plt.show()
        '''
        #INITAITE DICTIONARY 
        print('Calculating...')   
        data = {                                                                \
                'x':[x],                                                        \
                'total_energy':[total_energy(x, x_old, n_particles, dt)],       \
                'amps':[mode_energies(x, x_old, n_particles, dt)]
                }
        data['time'] = np.linspace(0, len(data['x']), num=len(data['x'])) * dt
        #convert to numpy-array:
        for key in data.keys():
            data[key] = np.array(data[key])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(data['amps'])
        e_crit = data['amps'][0][15]
        b = n_particles*np.sqrt(1)/(mod*e_crit)
        betas.append(b)
        e_crits.append(e_crit)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creates Plot for Energies vs Time
    f,ax = plt.subplots()
    f.tight_layout()
    print(betas)
    ax.plot(betas, e_crits, c='k')
    #ax.set_ylim((0,.01))
    #ax.set_xlim((0,.02))
    ax.set_xlabel(r'$\beta$',fontsize=14)
    ax.set_ylabel(r'$\epsilon_{crit}$',fontsize=14)
    ax.set_title(r'$\epsilon_{crit}$'+' vs '+r'$\beta$',fontsize=16)
    ax.legend(loc="upper right")
    f.show()
    f.savefig("Q3a.png", bbox_inches='tight',dpi=600)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nparticles = [16,17,18,19,20,21,22,23,24]
    e2 = []
    for n in nparticles:
        init_mode = mod
        amplitude = np.sqrt(2/(n+1))
        x = x_old = [0.0 for i in list(range(0,n+2))]
        tmp = np.arange(0, n+2, 1)
        x = x_old = 10*amplitude*np.sin(init_mode*np.pi*tmp/(n+1))
        x[0] = x[-1] = x_old[0] = x_old[-1] = 0
        #PLOT INITIAL POSITIONS
        '''
        pos = np.linspace(0, n + 2, num=n + 2)
        plt.title('Start position of particles')
        plt.plot(pos, x, 'ro')
        plt.show()
        '''
        #INITAITE DICTIONARY 
        print('Calculating...')   
        data = {                                                                \
                'x':[x],                                                        \
                'total_energy':[total_energy(x, x_old, n, dt)],       \
                'amps':[mode_energies(x, x_old, n, dt)]
                }
        #LOOP THROUGH ALL TIMES ~300 PERIODS
        data['time'] = np.linspace(0, len(data['x']), num=len(data['x'])) * dt
        #convert to numpy-array:
        for key in data.keys():
            data[key] = np.array(data[key])
        e_crit = data['amps'][0][15]
        e2.append(e_crit)
    print(nparticles,e2)
    f1,ax1 = plt.subplots()
    f1.tight_layout()    
    ax1.scatter(nparticles,e2, c='k')
    #ax1.set_ylim((0,.01))
    ax1.set_xlabel('Number of Particles',fontsize=14)
    ax1.set_ylabel(r'$\epsilon_{crit}$',fontsize=14)
    ax1.set_title(r'$\epsilon_{crit}$'+' vs '+'Number of Particles for '+r'$\ell=16$',fontsize=16)
    ax1.legend(loc="upper right")
    f1.show()
    f1.savefig("Q3b.png", bbox_inches='tight',dpi=600)
    
###############################################################################
# MAIN CODE
###############################################################################
#QuestionOne(1,2)
#QuestionOne(.45,3)
#QuestionThree(16)
#QuestionOne(0,1)
QuestionOne(1,1)
#QuestionOne(.3,1)
#QuestionOne(3,1)
#QuestionTwo(.01,16)
#QuestionTwo(.02,16)
#QuestionTwo(.03,16)
#QuestionTwo(1,1)
#QuestionTwo(3,1)
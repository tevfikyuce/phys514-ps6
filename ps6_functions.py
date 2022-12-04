import numpy as np
import scipy.constants
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time

#Defining and calculating constants of scaled equations
#Known physical constants
G = scipy.constants.G
rp = 7.405e11
ms = 1.988e30
mJ = 1.898e27
vp = 13.72e3

#Calculated scale constants
t0 = np.sqrt(np.power(rp, 3)/(G*(ms+mJ)))
mu = (ms*mJ)/(ms+mJ)
mu_S = ms/mu
mu_J = mJ/mu
vJ_initial = vp*t0/rp

def calculate_rhs(vec):
    rhs = np.zeros(len(vec)) #Initialize rhs as a zero vector

    #Position vector of Jupyter and Sun
    rhoJ = np.array([vec[0], vec[1]]).astype(float)
    rhoS = np.array([vec[2], vec[3]]).astype(float)

    #Displacement vector between Jupyter and Sun
    dispJ = rhoS-rhoJ
    dispS = rhoJ-rhoS

    dispJ_unit = dispJ/np.linalg.norm(dispJ)
    dispS_unit = dispS/np.linalg.norm(dispS)

    #Time derivative of Jupyter's x position
    rhs[0] = vec[4]
    #Time derivative of Jupyter's y poisiton
    rhs[1] = vec[5]
    #Tİme derivative of Sun's x position
    rhs[2] = vec[6]
    #Time derivative of Sun's y position
    rhs[3] = vec[7]
    #Time derivative of Jupyter's x velocity
    rhs[4] = (1/mu_J)*(1/np.power(np.linalg.norm(dispJ), 2))*dispJ_unit[0]
    #Time derivative of Jupyter's y velocity
    rhs[5] = (1/mu_J)*(1/np.power(np.linalg.norm(dispJ), 2))*dispJ_unit[1]
    #Time derivative of Sun's x velocity
    rhs[6] = (1/mu_S)*(1/np.power(np.linalg.norm(dispS), 2))*dispS_unit[0]
    #Time derivative of Sun's y velocity
    rhs[6] = (1/mu_S)*(1/np.power(np.linalg.norm(dispS), 2))*dispS_unit[1]

    return rhs

def time_step(vec, t, rhs, dt, method):
    def implicit_euler_f(x):
        return vec + dt*calculate_rhs(x) - x

    if method == 'forward-euler':
        return vec + dt*rhs
    elif method == 'implicit-euler':
        return fsolve(implicit_euler_f, vec)
    elif method == 'RK4':
        f1 = calculate_rhs(vec)
        f2 = calculate_rhs(vec + (dt/2)*f1)
        f3 = calculate_rhs(vec + (dt/2)*f2)
        f4 = calculate_rhs(vec + dt*f3)
        return vec + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)
    elif method == 'symplectic-euler':
        #First calculate with explicit euler
        next_vec = vec + dt*rhs
        #Now integrate position with using explicitly calculated vector
        next_vec[0:4] = vec[0:4] + dt*next_vec[4:]
        return next_vec
    else:
        print('Invalid method selected !!!')
        return None

def run_simulation(t_initial, t_final, dt, vec0, method):
    t = t_initial
    N = len(vec0) #Number of variables
    #Initialize the time vector
    t_arr = np.empty([1, 1])
    t_arr[:,0] = t_initial 
    #Initialize the vector that holds calculated variables
    x_arr = np.empty([N, 1])
    x_arr[:, 0] = vec0 

    start_time = time.time()
    while t<=t_final:
        #Get current vector
        vec = x_arr[:, -1]
        #vec = np.reshape(vec, (N,))
        #Iterate the time
        x_next = time_step(vec = vec, t = t, rhs=calculate_rhs(vec), dt=dt, method=method)
        x_next = np.reshape(x_next, (N,1))
        #Add next x and t to matrix
        x_arr = np.concatenate((x_arr, x_next), axis=1)
        t_arr = np.concatenate((t_arr, np.reshape(t, (1,1))), axis=1)

        t += dt
    end_time = time.time()

    print('Simulation finished !!!')
    return x_arr, t_arr, end_time-start_time

def calc_linear_momentum(x_arr):
    x_momentum = x_arr[4,:]*mu_J + x_arr[6,:]*mu_S #Linear momentum in x-direction
    y_momentum = x_arr[5,:]*mu_J + x_arr[7,:]*mu_S #Linear momentum in y-direction

    return np.sqrt(np.power(x_momentum, 2) + np.power(y_momentum, 2))

def calculate_energy(x_arr):
    distance_square = np.power(x_arr[0,:] - x_arr[2, :], 2) + np.power(x_arr[1,:] - x_arr[3,:], 2) #Square of the distance between two masses
    grav_potential = -1/distance_square #Gravitational potential
    kinetic_energy = 0.5*mu_J*(np.power(x_arr[4, :], 2) + np.power(x_arr[5, :], 2)) + 0.5*mu_S*(np.power(x_arr[6, :], 2) + np.power(x_arr[7, :], 2))

    return np.squeeze(grav_potential + kinetic_energy)

def calculate_angular_momentum(x_arr):
    #Calculate angular momentum of Jupyter around Sun
    r = x_arr[0:2,:] - x_arr[2:4,:] #Position Vector
    p = mu_J * x_arr[4:6] #Momentum Vector

    #Iterate over each time moment
    (N, n_sample) = x_arr.shape
    L_arr = np.zeros(n_sample)
    for i in range(n_sample):
        temp_L = np.cross(r[:,i], p[:,i])
        L_arr[i] = np.linalg.norm(temp_L)
    
    return L_arr

def plot_sim_results(t_arr, x_arr, method, dt, sim_time):
    
    t_arr = np.squeeze(t_arr) #Change number of dimensions in t array
    E = calculate_energy(x_arr=x_arr) #Total energy in the system
    p = calc_linear_momentum(x_arr=x_arr) #Total Linear Momentum in the system
    L = calculate_angular_momentum(x_arr=x_arr) #Total angular momentum in the system

    #Plot the results
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(15,9)
    fig.suptitle('Simulation results for ' + method + ' h = ' + str(dt) + ' t = ' + str(sim_time), fontweight='bold', fontsize=16)
    #First plot trajectories
    axs[0,0].plot(x_arr[0,:], x_arr[1,:])
    axs[0,0].plot(x_arr[2,:], x_arr[3,:])
    axs[0,0].set_xlabel('x')
    axs[0,0].set_ylabel('y')
    axs[0,0].legend(['Jupyter', 'Sun'])
    axs[0,0].set_title('Trajectroies of Jupyter and Sun')

    #Plot Energy vs Time
    axs[0,1].plot(t_arr, E)
    axs[0,1].set_xlabel('t')
    axs[0,1].set_ylabel('E')
    axs[0,1].set_title('Energy vs Time')

    #Plot Linear Momentum vs Time
    axs[1,0].plot(t_arr, p)
    axs[1,0].set_xlabel('t')
    axs[1,0].set_ylabel('p')
    axs[1,0].set_title('Linear Momentum vs Time')

    #Plot Angular Momentum vs Time
    axs[1,1].plot(t_arr, L)
    axs[1,1].set_xlabel('t')
    axs[1,1].set_ylabel('L')
    axs[1,1].set_title('Angular Momentum vs Time')

def three_body_acc(x, x_J, x_S):
    #This function calculates acceleration
    r_J2 = np.power(x[0]-x_J[0], 2) + np.power(x[1]-x_J[1], 2) #Distance squared to Jupyter
    r_S2 = np.power(x[0]-x_S[0], 2) + np.power(x[1]-x_S[1], 2) #Distance squared to Sun

    #Unit vector to the Jupyter
    r_J = x_J - x
    r_J = r_J/np.linalg.norm(r_J)

    #Unit vector to the Sun
    r_S = x_S - x
    r_S = r_S/np.linalg.norm(r_S)

    a = (1/(mu_S))*np.power(r_J2, -1)*r_J + (1/(mu_J))*np.power(r_S2, -1)*r_S
    return a

def transform_coords_to_rotating(t_arr, x_arr, x0, omega):
    n_points = len(t_arr)
    rotated_x = np.zeros(x_arr.shape, dtype=float)

    for i in range(n_points):
        t = t_arr[i] #Current time
        rotate_matrix = np.asmatrix([[np.cos(omega*t), np.sin(omega*t)], [-np.sin(omega*t), np.cos(omega*t)]]).astype(float) #Rotation matrix
        x = x_arr[:,i] #Current position
        rotated_coord = rotate_matrix@x - x0 #Transformed position vector
        rotated_x[:, i] = rotated_coord

    return rotated_x

def restricted_three_body_simulate(x0, v0, R_j, R_s, omega, t_final, dt):
    #Initial positions of Jupyter and Sun
    x_J0 = np.asarray([R_j, 0]).astype(float)
    x_S0 = np.asarray([R_s, 0]).astype(float)

    #First and initial points of data
    x_arr = np.zeros([2, 2])
    x_arr[:, 0] = x0 #Initial postion
    x_arr[:, 1] = x0 + v0*dt +(1/2)*three_body_acc(x0, x_J0, x_S0)*np.power(dt, 2)

    t_arr = [0, dt]
    t = dt

    while t<= t_final:
        t = t_arr[-1] #Load current time
        xJ = np.asarray([R_j*np.cos(omega*t), R_j*np.sin(omega*t)]).astype(float) #Current position of Jupyter
        xS = np.asarray([R_s*np.cos(omega*t), R_s*np.sin(omega*t)]).astype(float) #Current position of Sun
        current_x = x_arr[:, -1] #Current position of the object

        a = three_body_acc(current_x, xJ, xS) #Acceleration
        x_next = 2 * current_x - x_arr[:, -2] + np.power(dt, 2) * a #Calculate next position
        x_next = np.reshape(x_next, (2,1))

        t += dt

        x_arr = np.concatenate((x_arr, x_next), axis=1) #Add next position to array
        t_arr.append(t) #Add next time

    #Transformation of coordinates
    rotated_coords = transform_coords_to_rotating(t_arr=t_arr, x_arr=x_arr, x0=x0, omega=omega)

    return t_arr, rotated_coords

def time_step_pert_harmonic(x, t, dt, f):
    f1 = f(x)
    f2 = f(x + (dt/2)*f1)
    f3 = f(x + (dt/2)*f2)
    f4 = f(x + dt*f3)
    return x + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)

def run_pert_harmonic(omega, A, l, dt, t_final, show_potential=False):
    #Plotting potential
    def plot_potential(w, l, A, m, x_end):
        x = np.linspace(-np.abs(x_end), np.abs(x_end), 1000)

        V = (1/2)*m*np.power(w, 2)*np.power(x, 2) + m*(w*w)*A*np.exp(-np.power(x, 2)/(2*np.power(l, 2)) )

        plt.figure(figsize=(9,5))
        plt.plot(x, V)
        plt.xlabel('x')
        plt.ylabel('V(x)')
        plt.title('Perturbed Harmonic Potential')
        plt.show()

    #Define RHS of ODE
    def rhs_pert_harmonic(x):
        f = np.zeros(2)
        f[0] = x[1]
        f[1] = -x[0] + (A/np.power(l, 2))*x[0]*np.exp(-np.power(x[0], 2)/(2*np.power(l, 2)))
        return f

    #Plotting Potential
    if show_potential:
        plot_potential(w=omega, l=l, A=A, m=1, x_end=2)

    #Initial Vector
    x0 = np.asarray([1, 0]).astype(float)

    #Running Simulation
    t_arr = np.zeros(1)
    x_arr = np.zeros((2,1))
    x_arr[:, -1] = x0
    t = 0

    while t <= t_final:
        t = t_arr[-1] #Current t
        x = x_arr[:,-1] #Current x

        x_next = time_step_pert_harmonic(x=x, t=t, dt=dt, f=rhs_pert_harmonic)
        x_next = np.reshape(x_next, (2,1))
        
        t += dt
        t_arr = np.concatenate((t_arr, [t]))
        x_arr = np.concatenate((x_arr, x_next), axis=1)

    #Visualization of Simulation Result
    plt.figure(figsize=(12,7))
    plt.plot(t_arr, x_arr[0,:])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Position vs Time for Perturbed Harmonic Potential')
    plt.show()
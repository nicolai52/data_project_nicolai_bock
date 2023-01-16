# this is the imports needed to run reste of the script
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
st.code("""# this is the imports needed to run reste of the script
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp""", language="python")

st.markdown("""nicolai Bock wgz913
""", unsafe_allow_html=False)


#a couple of helper functions
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def make_phase_plot(solutions, title =  "phase plot"):

    fig,ax = plt.subplots()

    for init, solution in solutions.items():
        x,v,t = solution
        ax.plot(x, v, label = f'{init=}')

    ax.set_title(title)
    ax.set_ylabel('x [m]')
    if not isnotebook():
      
   
        ax.set_xlabel(r"v [m/s]")
        fig.legend()
        st.pyplot(fig)
    else:
        ax.set_xlabel(r"v [$\frac{m}{s}$]")
        plt.legend()
        plt.show()

def convert_ivpt_dict_to_eluer_solver(ivpt_dict):
    new_solution = {}

    for key, value in ivpt_dict.items():
        new_solution[key] = np.append(value.y,[value.t],axis=0)
    return new_solution

def make_plot(solutions,title =  "displacement in x"):
    fig,ax = plt.subplots()

    for dt,solution in solutions.items():
        x,v,t = solution
        ax.plot(t,x,label=f'{dt=}')
    ax.set_title(title)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')

    

    if not isnotebook():
        fig.legend()
        st.pyplot(fig)
    else:
        plt.legend(loc='upper right')
        plt.show()

st.code("""

#a couple of helper functions
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def make_phase_plot(solutions, title =  "phase plot"):

    fig,ax = plt.subplots()

    for init, solution in solutions.items():
        x,v,t = solution
        ax.plot(x, v, label = f'{init=}')

    ax.set_title(title)
    ax.set_ylabel('x [m]')
    if not isnotebook():
      
   
        ax.set_xlabel(r"v [m/s]")
        fig.legend()
        st.pyplot(fig)
    else:
        ax.set_xlabel(r"v [$\frac{m}{s}$]")
        plt.legend()
        plt.show()

def convert_ivpt_dict_to_eluer_solver(ivpt_dict):
    new_solution = {}

    for key, value in ivpt_dict.items():
        new_solution[key] = np.append(value.y,[value.t],axis=0)
    return new_solution

def make_plot(solutions,title =  "displacement in x"):
    fig,ax = plt.subplots()

    for dt,solution in solutions.items():
        x,v,t = solution
        ax.plot(t,x,label=f'{dt=}')
    ax.set_title(title)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('x [m]')

    

    if not isnotebook():
        fig.legend()
        st.pyplot(fig)
    else:
        plt.legend(loc='upper right')
        plt.show()
""", language="python")

st.markdown("""# Question 1""", unsafe_allow_html=False)

st.markdown("""## a 1)
her i implementation the function for solving the ode""", unsafe_allow_html=False)

init = np.array([1,0])
tmax  = 50
dt = 0.1

def ode(z: tuple[float,float],k,g) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -omega*x])

def simulate_euler(init,*args,max = tmax, dt = dt, ode = ode) -> np.ndarray:
    
    time_step_over_range = np.arange(0,tmax,dt)
    sol = [init]

    for _ in time_step_over_range[0:-1]:
        next_sol = sol[-1] + ode(sol[-1],*args) * dt
        
        sol.append(next_sol)

    sol = np.array(sol)

    return np.append(sol.T, np.array([time_step_over_range]), axis = 0)



st.code("""
init = np.array([1,0])
tmax  = 50
dt = 0.1

def ode(z: tuple[float,float],k,g) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -omega*x])

def simulate_euler(init,*args,max = tmax, dt = dt, ode = ode) -> np.ndarray:
    
    time_step_over_range = np.arange(0,tmax,dt)
    sol = [init]

    for _ in time_step_over_range[0:-1]:
        next_sol = sol[-1] + ode(sol[-1],*args) * dt
        
        sol.append(next_sol)

    sol = np.array(sol)

    return np.append(sol.T, np.array([time_step_over_range]), axis = 0)


""", language="python")

st.markdown("""## a 2)""", unsafe_allow_html=False)
dts = [0.1,0.01,0.001]
solutions_dts  =  {}

for _dt in dts:
    solutions_dts[_dt] = simulate_euler(init,1,1, dt = _dt)
make_plot(solutions_dts)
st.code("""dts = [0.1,0.01,0.001]
solutions_dts  =  {}

for _dt in dts:
    solutions_dts[_dt] = simulate_euler(init,1,1, dt = _dt)
make_plot(solutions_dts)""", language="python")
inits = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
dt = 0.001
solutions  =  {}

for _init in inits:
    solutions[_init[0]] = simulate_euler(_init,1,1, dt=dt)
make_plot(solutions)
st.code("""inits = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
dt = 0.001
solutions  =  {}

for _init in inits:
    solutions[_init[0]] = simulate_euler(_init,1,1, dt=dt)
make_plot(solutions)""", language="python")


def ode1(t,z: tuple[float,float],k,g) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -omega*x**3])

inits = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
solutions_ivp  =  {}

for _init in inits:
    solutions_ivp[_init[0]] = solve_ivp(ode1, [0,50],_init, args = [1,1],max_step = 0.01)


## this is bade pratice to change to type of a variable but it will this time :))
solutions_ivp = convert_ivpt_dict_to_eluer_solver(solutions_ivp)
make_plot(solutions_ivp)

st.code("""

def ode1(t,z: tuple[float,float],k,g) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -omega*x**3])

inits = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
solutions_ivp  =  {}

for _init in inits:
    solutions_ivp[_init[0]] = solve_ivp(ode1, [0,50],_init, args = [1,1],max_step = 0.01)


## this is bade pratice to change to type of a variable but it will this time :))
solutions_ivp = convert_ivpt_dict_to_eluer_solver(solutions_ivp)
make_plot(solutions_ivp)
""", language="python")
make_phase_plot(solutions)
st.code("""make_phase_plot(solutions)""", language="python")
def ode3(z: tuple[float,float],k,g,mu) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -mu*(x**2-1)*v-omega**2*x])

inits_3 = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
dt = 0.01
solutions_3  =  {}

for _init in inits_3:
    
    solutions_3[_init[0]] = simulate_euler(_init,1,1,1, dt=dt, ode= ode3)


make_phase_plot(solutions_3)
make_plot(solutions_3)

st.code("""def ode3(z: tuple[float,float],k,g,mu) -> np.ndarray:
    omega =  np.sqrt(k/g)
    x,v = z
    return np.array([v, -mu*(x**2-1)*v-omega**2*x])

inits_3 = [np.array([1,0]),np.array([2,0]),np.array([5,0])]
dt = 0.01
solutions_3  =  {}

for _init in inits_3:
    
    solutions_3[_init[0]] = simulate_euler(_init,1,1,1, dt=dt, ode= ode3)


make_phase_plot(solutions_3)
make_plot(solutions_3)
""", language="python")
solutions_3_20 = {}
for key, solution in solutions_3.items():
    x,v,t = solution
    mask = t <= 20
    solutions_3_20[key] = np.array([x[mask],v[mask],t[mask]])
make_phase_plot(solutions_3_20)
solutions_3_400 = {}
for key, solution in solutions_3.items():
    x,v,t = solution
    mask = t >= 400
    solutions_3_400[key] = np.array([x[mask],v[mask],t[mask]])
make_phase_plot(solutions_3_400)
st.code("""solutions_3_20 = {}
for key, solution in solutions_3.items():
    x,v,t = solution
    mask = t <= 20
    solutions_3_20[key] = np.array([x[mask],v[mask],t[mask]])
make_phase_plot(solutions_3_20)
solutions_3_400 = {}
for key, solution in solutions_3.items():
    x,v,t = solution
    mask = t >= 400
    solutions_3_400[key] = np.array([x[mask],v[mask],t[mask]])
make_phase_plot(solutions_3_400)""", language="python")
ode3_ivp =  lambda t, *args, **kwargs: ode3(*args, **kwargs)
solutions_3_ivp  =  {}

for _init in inits_3:
    solutions_3_ivp[_init[0]] = solve_ivp(ode3_ivp, [0,tmax],_init, args = [1,1,1],max_step = 0.01)
solutions_3_ivp = convert_ivpt_dict_to_eluer_solver(solutions_3_ivp)
make_phase_plot(solutions_3_ivp)
make_plot(solutions_3_ivp)

st.code("""ode3_ivp =  lambda t, *args, **kwargs: ode3(*args, **kwargs)
solutions_3_ivp  =  {}

for _init in inits_3:
    solutions_3_ivp[_init[0]] = solve_ivp(ode3_ivp, [0,tmax],_init, args = [1,1,1],max_step = 0.01)
solutions_3_ivp = convert_ivpt_dict_to_eluer_solver(solutions_3_ivp)
make_phase_plot(solutions_3_ivp)
make_plot(solutions_3_ivp)
""", language="python")
ode3_ivp =  lambda t, *args, **kwargs: ode3(*args, **kwargs)
solutions_3_ivp_mu  =  {}

for _init in inits_3:
    solutions_3_ivp_mu[_init[0]] = solve_ivp(ode3_ivp, [0,tmax],_init, args = [1,1,7],max_step = 0.01)
solutions_3_ivp_mu = convert_ivpt_dict_to_eluer_solver(solutions_3_ivp_mu)
make_phase_plot(solutions_3_ivp_mu)
make_plot(solutions_3_ivp_mu)

st.code("""ode3_ivp =  lambda t, *args, **kwargs: ode3(*args, **kwargs)
solutions_3_ivp_mu  =  {}

for _init in inits_3:
    solutions_3_ivp_mu[_init[0]] = solve_ivp(ode3_ivp, [0,tmax],_init, args = [1,1,7],max_step = 0.01)
solutions_3_ivp_mu = convert_ivpt_dict_to_eluer_solver(solutions_3_ivp_mu)
make_phase_plot(solutions_3_ivp_mu)
make_plot(solutions_3_ivp_mu)
""", language="python")

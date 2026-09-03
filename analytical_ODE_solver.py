# Analytical ODE solver for an ODE defined by y_dotdot + A(y_dot) + B(y) = C.
# dt = the step size desired by the user
# y0 = y value at the starting time
# y_dot0 = y_dot value at the starting time
# t0 = starting time of desired analysis
# t1 = ending time of desired analysis

def analytical_ODE_solver(A, B, C, dt, y0, y_dot0, t0, t1):
    import numpy as np
    omega = np.sqrt(B-((A**2)/4))

    # Make an array for each time step (this is an array of time values)
    t = np.arange(t0, t1, dt)

    # Make arrays for y and y_dot (this is an array of y position and its derivative)
    y = np.zeros(len(t))
    y_dot = np.zeros(len(t))

    # Set initial conditions (this is the first value in the array or the first time step)
    y[0] = y0
    y_dot[0] = y_dot0

    # Calculate the analytical solution for each time step
    for n in range(0, len(t)-1):
        y[n+1] = (C/B) + np.exp(-A*dt/2)*( (y[n]-(C/B))*np.cos(omega*dt) + (1/omega)*(y_dot[n]+(A/2)*(y[n]-(C/B)))*np.sin(omega*dt) ) 
        y_dot[n+1] = np.exp(-A*dt/2)*( y_dot[n]*np.cos(omega*dt) - (1/omega)*( (A/2)*y_dot[n] + B*(y[n]-(C/B)) )*np.sin(omega*dt) )

    return t, y, y_dot
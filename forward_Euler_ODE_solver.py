# Forward Euler method, 1st order from Ch 16 pg 5
# A, B, and C are constants such that y_dotdot + A*y_dot + B*y = C
# dt = 'h' in textbookk, the timestep
# y0 = y value at the starting time
# y_dot0 = y_dot value at the starting time
# t0 = starting time of desired analysis
# t1 = ending time of desired analysis

def forward_Euler_ODE_solver(A, B, C, dt, y0, y_dot0, t0, t1):
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
        y[n+1] = y[n] + dt*y_dot[n]
        y_dot[n+1] = y_dot[n] + dt*( C - A*y_dot[n] - B*y[n] )

    return t, y, y_dot
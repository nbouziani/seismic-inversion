# Command to run:
#    `python seismic_inversion.py -regulariser {regulariser} -scale_noise {noise scaling factor} -alpha {regularisation factor}`
#
# List of regularisers:
#    0: No regularisation
#    1: Tikhonov
#    2: Neural network regularisation using an ExternalOperator
#
# Run command to reproduce the article figures: `python seismic_inversion.py -regulariser 0 1 2`

import os
from zipfile import ZipFile

from firedrake import *
from firedrake_adjoint import *

import numpy as np
import matplotlib.pyplot as plt
import argparse


# -- Parsing and model extraction -- #

# Extract the saved model
if not os.path.exists('./models/model.pth'):
    with ZipFile('./models/model.zip', 'r') as z:
        z.extractall('./models/')
    print('\n Extraction of the saved model: completed!\n')

# Retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument('-regulariser', nargs='+', type=int, default=[0])
parser.add_argument('-scale_noise', type=float, default=0.1)
parser.add_argument('-alpha', type=float, default=0.5)
args = parser.parse_args()

# Display summary
print('\n --- Hyperparameters --- \n')
for k, v in args._get_kwargs():
    print(k, ' : ', v)
print('\n List of regularisers: ',
      '\n\t 0: No regularisation',
      '\n\t 1: Tikhonov',
      '\n\t 2: Neural network regularisation using an ExternalOperator')

list_regulariser = set(args.regulariser)
if any(i not in (0, 1, 2) for i in list_regulariser):
    raise ValueError('Regulariser index must be in (0, 1, 2)!')


# -- Seismic inversion-- #

def seismic_inversion(regulariser, scale_noise, alpha):
    r""" Compute the solution of the inverse problem

    :arg regulariser: An integer indicating the type of regularisation to take into account.
    :arg scale_noise: Scale factor applied on the noise to make the observed data from the exact solution.
    :arg alpha: Regularisation factor

    :returns: The solution computed for the given regulariser as well as the exact solution.

    """

    # Set mesh and function spaces
    Lx = 5
    Ly = 5
    mesh = RectangleMesh(100, 100, Lx, Ly)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    P = FunctionSpace(mesh, 'DG', 0)

    # Functions
    u = TrialFunction(V)
    v = TestFunction(V)
    vel = Function(P)

    # Source term
    x0 = Lx / 2
    y0 = Ly
    f = Function(V).interpolate(exp(- 120 * ((x - x0)**2 + (y - y0)**2)))

    # Exact velocity model
    c_exact = Function(P).interpolate(conditional((x - Lx / 2)**2 + (y - Ly / 2)**2 < 1, 2, 1))

    # Define the forward problem: solve the PDE for a given velocity model
    def F(vel):
        T = 1.
        dt = 0.01
        t = 0

        p = Function(V, name="p")
        phi = Function(V, name="phi")
        while t <= T:
            phi -= dt / 2 * p

            rhs = v * p * dx + dt * inner(grad(v), vel**2 * grad(phi)) * dx
            if t <= 10 * dt:
                rhs += inner(f, v) * dx
            solve(u * v * dx == rhs, p, solver_parameters={'ksp_type': 'gmres'})
            phi -= dt / 2 * p
            t += dt
        return phi

    # Build observed data by adding noise
    phi_exact = F(c_exact)
    noise = scale_noise * np.random.rand(V.node_count)
    phi_exact.dat.data[:] += noise

    # Load the neural network
    import torch
    from models.model import NeuralNet
    model = torch.load('./models/model.pth')
    model.double()

    # Define the ExternalOperator
    p = neuralnet(model, function_space=P)

    # Forward simulation
    phi = F(vel)

    # Set the input
    N = p(vel)

    if regulariser == 0:
        print('\n --- Problem without regularisation --- \n')
        J = assemble(0.5 * (inner(phi - phi_exact, phi - phi_exact)) * dx)
        Jhat = ReducedFunctional(J, Control(vel))
        c_opt = minimize(Jhat, method="L-BFGS-B", tol=1.0e-10, options={"disp": True, "maxiter": 20})
    elif regulariser == 1:
        print('\n --- Problem with Tikhonov regularisation --- \n')
        J = assemble(0.5 * (inner(phi - phi_exact, phi - phi_exact) + alpha * inner(vel, vel)) * dx)
        Jhat = ReducedFunctional(J, Control(vel))
        c_opt = minimize(Jhat, method="L-BFGS-B", tol=1.0e-10, options={"disp": True, "maxiter": 20})
    elif regulariser == 2:
        print('\n --- Problem with neural network regularisation --- \n')
        J = assemble(0.5 * (inner(phi - phi_exact, phi - phi_exact) + alpha * inner(N, N)) * dx)
        Jhat = ReducedFunctional(J, Control(vel))
        c_opt = minimize(Jhat, method="L-BFGS-B", tol=1.0e-10, options={"disp": True, "maxiter": 20})
    else:
        raise ValueError('Regulariser index must be in (0, 1, 2)!')

    return c_opt, c_exact


# -- Plots -- #

def save_fig(c, name):
    fig = tripcolor(c)
    # plt.title('Velocity model: ' + name)
    plt.axis('off')
    plt.colorbar(fig, shrink=0.907)
    name = name.lower()
    name = name.replace(' ', '_')
    plt.savefig('./figures/seismic_inversion_' + name + '.png', bbox_inches='tight')


map_title = {0: 'without regularisation', 1: 'Tikhonov regularisation', 2: 'NN regularisation'}
for regulariser in list_regulariser:
    c_opt, c_exact = seismic_inversion(regulariser, args.scale_noise, args.alpha)
    save_fig(c_opt, map_title[regulariser])
save_fig(c_exact, 'exact')
plt.show()

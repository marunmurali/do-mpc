#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('/home/arun/do-mpc')
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(5,1))
    #[x, y, theta, v, alpha]

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))
    #[v, alpha]

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    #model.set_expression(expr_name='cost', expr=sum1(_x**2))
    model.set_expression(expr_name='cost', expr=sum1(((_x[1]**2+_x[0]**2)-1)**2+(_x[3]-5)**2))
    
    #time interval
    dt = 0.02
    
    #print(xtemp)
    A = np.array([[ 1,  0,  0,  0.127*dt*cos(_x[2])*cos(_x[4]), 0],
                  [ 0,  1,  0,  0.127*dt*sin(_x[2])*cos(_x[4]), 0],
                  [ 0,  0,  1,  dt*sin(_x[4]), 0],
                  [ 0,  0,  0,  0.9661, 0],
                  [ 0,  0,  0,  0, 0]])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0.0315, 0],
                  [0, 1]])


    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)

    model.setup()

    return model

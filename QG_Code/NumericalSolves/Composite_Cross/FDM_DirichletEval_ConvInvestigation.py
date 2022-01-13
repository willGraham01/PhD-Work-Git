#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:29:54 2022

@author: will

The cross-in-plane geometry, and our Helmholtz equation wrt the composite measure on it, is satisfied by a subset of the eigenfunctions of the Dirichlet Laplacian on $\Omega$.

Observe that, for $\Omega=(0,1)^2$ and $\theta\in[-\pi,\pi)^2$, the eigenpairs of the Dirichlet Laplacian which solve the equation

\begin{align*}
    -\Delta_{\theta}u = \lambda u, \qquad u\vert_{\partial\Omega} = 0,
\end{align*}

are given by 

\begin{align*}
    u_{nm}(x,y) &= \mathrm{e}^{-\mathrm{i}\theta\cdot\mathbf{x}}\sin\left(n\pi x\right)\sin\left(m\pi y\right), \\
    \lambda_{nm} &= \left( n^2 + m^2 \right)\pi^2.
\end{align*}

In the event that both of the following hold:
- ( $n$ is even and $\theta_1=0$ ) or ( $n$ is odd and $\theta_1=-\pi$ ),
- ( $m$ is even and $\theta_2=0$ ) or ( $m$ is odd and $\theta_2=-\pi$ ),

then $u_{nm}$ and $\lambda_{nm}$ are also an eigenpair for our Helmholtz equation with respect to the composite measure.

This script looks to test whether the FDM solver is getting close to one of these eigenfunctions.
Specifically, we will look to test whether as we increase $N$ (the number of gridpoints used), the difference in the nearest eigenvalue and eigenfunction gradually gets closer to the analytic solution when $n=m=1$.

This provides us with the eigenfunction and eigenvalue

\begin{align*}
    U(x,y) &= \mathrm{e}^{\mathrm{i}\pi(x+y)}\sin\left(\pi x\right)\sin\left(\pi y\right), \\
    \lambda =: \omega^2 &= 2\pi^2, 
	\qquad \implies \omega \approx 4.442882938.
\end{align*}

"""

import numpy as np

from CompMes_FDM import FDM_findEvals
from CompMes-AllAnalysis import 
#FDM_FindEvals(N, theta, alpha3, lOff=False, nEvals=3, sigma=1., checks=False, saveEvals=True, saveEvecs=False)
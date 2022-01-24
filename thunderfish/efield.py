"""Simulations of spatial electric fields.

## Electric monopoles

For simulating the spatial geometry of electric fields generated by electric fishes
and perturbed by objects, first generate monopoles and charges:

- `efish_monopoles()`: monopoles for simulating the electric field of an electric fish.
- `object_monopoles()`: monopoles for simulating a circular object.

## Potential, electric field, and field lines

- `epotential()`: simulation of electric field potentials.
- `epotential_meshgrid()`: simulation of electric field potentials on a mesh grid.
- `efield()`: simulation of electric field.
- `efield_meshgrid()`: simulation of electric field on a mesh grid.
- `projection()`: projection of electric field on surface normals.
- `fieldline()`: compute an electric field line.

## Visualization

- `squareroot_transform()`: square-root transformation keeping the sign.
- `plot_fieldlines()`: plot field lines with arrows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def efish_monopoles(pos=(0, 0), direction=(1, 0), size=10.0, bend=0, nneg=1):
    """Monopoles for simulating the electric field of an electric fish.

    This implements the model published in
    Chen, House, Krahe, Nelson (2005) "Modeling signal and background
    components of electrosensory scenes", J Comp Physiol A 191: 331-345

    Ten monopoles per unit size are uniformly distributed along the fish's body axis.
    The first (tail) nneg monopoles get negative charges that equal in sum
    the sum of the positive unit charges of the remaining (head) monopoles.
    The strength of the dipole increases linearly with fish size.

    Pass the returned monopole positions and charges on to the epotential() function
    to simulate the resulting electric field potentials, to the efield() function
    to simulate the electric field, or to object_monopoles() to add an object.

    Parameters
    ----------
    pos: tuple of floats
        Coordinates of the fish's position (its center).
        The number of elements in pos set the number of dimensions to be used.
    direction: tuple of floats
        Coordinates of a vector defining the orientation of the fish.
        Missing dimensions are filled in with zeros.
        Note: currently only rotations in the x-y plane are implemented.
    size: float
        Size of the fish. Per size unit 10 monopols are distributed along
        the fish's body axis.
    bend: float
        Bending angle of the fish's tail in degree.
    nneg: int
        Number of negative charges to be used. The remaining ones are positively charged.

    Returns
    -------
    poles: 2D array of floats
        Positions of the monopoles with n-dimensional coordinates
        as specified by the number of elements in pos.
    charges: array of floats
        The charge of each monopole.

    Example
    -------
    ```
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    poles1 = efish_monopoles(*fish1)
    ```
    """
    n = int(10*size)
    npos = n - nneg
    ppx = 0.1
    pos = np.asarray(pos)
    dirv = np.zeros(len(pos))
    dirv[:len(direction)] = direction        
    charges = np.ones(n)
    charges[:nneg] = -float(npos)/nneg
    poles = np.zeros((n, len(pos)))
    poles[:,0] = np.arange(-n//2, -n//2+n)*ppx
    if np.abs(bend) > 1.e-8:
        xm = -np.min(poles[:,0])       # tip of fish tail
        r = -180.0*xm/bend/np.pi       # radius of circle on which to bend the tail
        xp = poles[poles[:,0]<0.0,0]   # all negative x coordinates of poles
        beta = xp/r                    # angle on circle for each x coordinate
        poles[poles[:,0]<0.0,0] = -np.abs(r*np.sin(beta)) # transformed x coordinates
        poles[poles[:,0]<0.0,1] = r*(1.0-np.cos(beta))    # transformed y corrdinates
    # rotation matrix:
    theta = -np.arctan2(dirv[1], dirv[0])
    c = np.cos(theta)
    s = np.sin(theta)
    rm = np.array(((c, -s), (s, c)))
    # rotation:
    poles[:,:2] = np.dot(poles[:,:2], rm)
    # translation:
    poles += pos
    return poles, charges


def object_monopoles(pos=(0, 0), radius=1.0, chi=1.0, *args):
    """Monopoles for simulating a circular object.

    The circular object is approximated by an induced dipole, as
    sugested by Rasnow B (1996) "The effects of simple objects on the
    electric field of Apteronotus", J Comp Physiol A 178:397-411 and
    Chen, House, Krahe, Nelson (2005) "Modeling signal and background
    components of electrosensory scenes", J Comp Physiol A 191: 331-345.

    Pass the returned monopole positions and charges on to the
    epotential() function to simulate the resulting electric field
    potentials or to the efield() function to simulate the electric
    field.

    Two monopoles with charges q and -q separated by dx form a dipole
    with dipole moment p = q dx. According to Chen et al (2005), this
    dipole moment should equal chi*radius**3*E_obj, where E_obj is the
    electric field generated by the fishes at the position of the
    object. We normalize E_obj and multiply it with a small number eps
    to get dx. Accordingly, we have to set q to chi*radius**3
    |E_obj|/eps.

    Parameters
    ----------
    pos: tuple of floats
        Coordinates of the fish's position (its center).
        The number of dimensions must be the same as the one of the poles
        passed on in args.
    radius: float
        Radius of the small circular object.
    chi: float
        Electrical contrast. Unity for a perfect conductor, -0.5 for a
        perfect insulator and zero if the electrical impedance of the sphere
        matches that of the surrounding water.
    args: list of tuples
        Each tuple contains as the first argument the position of
        monopoles (2D array of floats), and as the second argument the
        corresponding charges (array of floats) of electric fish. Use
        efish_monopoles() to generate monopoles and corresponding charges.

    Returns
    -------
    poles: 2D array of floats
        Positions of the monopoles.
    charges: array of floats
        The charge of each monopole.

    Example
    -------
    ```
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = object_monopoles((-6, 0), 1.0, -0.5, poles1, poles2)
    ```
    """
    eps = 0.1   # distance of the two monopoles
    pos = np.asarray(pos)
    # electric field at object position:
    eobj = efield(pos, *args)
    eobjnorm = np.linalg.norm(eobj)
    # induced dipole:
    charges = np.ones(2)*chi*radius**3*eobjnorm/eps
    charges[0] = -charges[0]
    poles = np.zeros((2, len(pos)))
    poles[0,:] = -eobj*0.5*eps/eobjnorm   # distance between monopoles
    poles[1,:] = +eobj*0.5*eps/eobjnorm   # distance between monopoles
    poles += pos                          # translation to required position
    return poles, charges


def epotential(pos, *args):
    """Simulation of electric field potentials.

    Parameters
    ----------
    pos: 2D array of floats
        Each row contains the coordinates (2D or 3D)
        for which the potential should be computed.
    args: list of tuples
        Each tuple contains as the first argument the position of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    pot: 1D array of float
        The potential for each position in `pos`.
    """
    pos = np.asarray(pos)
    pot = np.zeros(len(pos))
    for poles, charges in args:
        for p, c in zip(poles, charges):
            r = pos - p
            rnorm = np.linalg.norm(r, axis=1)
            rnorm[np.abs(rnorm) < 1e-12] = 1.0e-12
            pot += c/rnorm
    return pot


def epotential_meshgrid(xx, yy, zz, *args):
    """Simulation of electric field potentials on a mesh grid.

    This is a simple wrapper for epotential().

    Parameters
    ----------
    xx: 2D array of floats
        Range of x coordinates as returned by numpy.meshgrid().
    yy: 2D array of floats
        Range of y coordinates as returned by numpy.meshgrid().
    zz: None or 2D array of floats
        z coordinates on the meshgrid defined by xx and yy.
        If provided, poles in args must be 3D.
        If None then treat it as a 2D problem with poles in args providing 2D coordinate.
    args: list of tuples
        Each tuple contains as the first argument the position (2D or 3D) of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    pot: 2D array of floats
        The potential for the mesh grid defined by xx and yy and evaluated
        at (xx, yy, zz).

    Example
    -------
    ```
    fig, ax = plt.subplots()
    maxx = 30.0
    maxy = 27.0
    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = object_monopoles((-6, 0), 1.0, -0.5, poles1, poles2)
    allpoles = (poles1, poles2, poles3)
    # potential:
    pot = epotential_meshgrid(xx, yy, None, *allpoles)
    thresh = 0.65
    zz = squareroot_transform(pot/200, thresh)
    levels = np.linspace(-thresh, thresh, 16)
    ax.contourf(x, y, -zz, levels, cmap='RdYlBu')
    ax.contour(x, y, -zz, levels, zorder=1, colors='#707070',
               linewidths=0.1, linestyles='solid')
    plt.show()
    ```
    """
    pos = np.vstack((xx.ravel(), yy.ravel())).T
    pot = epotential(pos, *args)
    return pot.reshape(xx.shape)

    
def efield(pos, *args):
    """Simulation of electric field given a set of electric monopoles.

    Parameters
    ----------
    pos: array of floats
        Each row contains the coordinates (2D or 3D)
        for which the potential should be computed.
        A single (1D) position is also accepted.
    args: list of tuples
        Each tuple contains as the first argument the position of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    field: array of floats
        The electric field components for each position in `pos`.
    """
    pos = np.asarray(pos)
    onedim = len(pos.shape) == 1
    if onedim:
        pos = pos.reshape(-1, len(pos))
    field = np.zeros(pos.shape)
    for poles, charges in args:
        for p, c in zip(poles, charges):
            r = pos - p
            rnorm = np.linalg.norm(r, axis=1)
            rnorm[np.abs(rnorm) < 1e-12] = 1.0e-12
            fac = c/rnorm**3
            field += r*fac[:,np.newaxis]
    return field[0] if onedim else field


def efield_meshgrid(xx, yy, zz, *args):
    """Simulation of electric field on a mesh grid.

    This is a simple wrapper for efield().
    
    Parameters
    ----------
    xx: 2D array of floats
        Range of x coordinates as returned by numpy.meshgrid().
    yy: 2D array of floats
        Range of y coordinates as returned by numpy.meshgrid().
    zz: None or 2D array of floats
        z coordinates on the meshgrid defined by xx and yy.
        If provided, poles in args must be 3D.
        If None then treat it as a 2D problem with poles in args providing 2D coordinate.
    args: list of tuples
        Each tuple contains as the first argument the position (2D or 3D) of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.

    Returns
    -------
    pot: 2D array of floats
        The potential for the mesh grid defined by xx and yy and evaluated
        at (xx, yy, zz).

    Returns
    -------
    fieldx: 2D array of floats
        The x-coordinate of the electric field for the mesh grid
        defined by xx and yy and evaluated at (xx, yy, zz).
    fieldy: 2D array of floats
        The y-coordinate of the electric field for the mesh grid
        defined by xx and yy and evaluated at (xx, yy, zz).
    fieldz: 2D array of floats
        The z-coordinate of the electric field for the mesh grid
        defined by xx and yy and evaluated at (xx, yy, zz).
        This is only returned if zz is not None.

    Example
    -------
    ```
    fig, ax = plt.subplots()
    maxx = 30.0
    maxy = 27.0
    x = np.linspace(-maxx, maxx, 40)
    y = np.linspace(-maxy, maxy, 40)
    xx, yy = np.meshgrid(x, y)
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = object_monopoles((-6, 0), 1.0, -0.5, poles1, poles2)
    allpoles = (poles1, poles2, poles3)
    fieldx, fieldy = efield_meshgrid(xx, yy, None, *allpoles)
    u = squareroot_transform(fieldx, 0)
    v = squareroot_transform(fieldy, 0)
    ax.quiver(qx, qy, u, v, units='xy', angles='uv', scale=2, scale_units='xy',
              width=0.07, headwidth=5)
    ``` 
    """
    if zz is None:
        pos = np.vstack((xx.ravel(), yy.ravel())).T
        ef = efield(pos, *args)
        return ef[:,0].reshape(xx.shape), ef[:,1].reshape(xx.shape)
    else:
        pos = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
        ef = efield(pos, *args)
        return ef[:,0].reshape(xx.shape), ef[:,1].reshape(xx.shape), ef[:,2].reshape(xx.shape)


def projection(ex, ey, ez, nx, ny, nz):
    """Projection of electric field on surface normals.

    Parameters
    ----------
    ex: array of floats
        x-coordinates of the electric field.
    ey: array of floats
        y-coordinates of the electric field.
    ez: array of floats
        z-coordinates of the electric field.
    nx: array of floats
        x-coordinates of the surface normals.
    ny: array of floats
        y-coordinates of the surface normals.
    nz: array of floats
        z-coordinates of the surface normals.
    """
    ef = np.vstack((ex.ravel(), ey.ravel(), ez.ravel())).T
    nf = np.vstack((nx.ravel(), ny.ravel(), nz.ravel())).T
    proj = np.sum(ef*nf, axis=1)
    return proj.reshape(ex.shape)


def fieldline(pos0, bounds, *args, eps=0.1, maxiter=1000):
    """Compute an electric field line.

    From the initial position `pos0` the field line is computed in both directions
    until it leaves the area defined by `bounds`.

    Parameters
    ----------
    pos0: array of floats
        Initial position for computing the field line.
    bounds: None or 2D array of floats
        If not None, stop integration of the field line if it exceeds bounds.
        First row are the minimum coordinates and second row the maximum coordinates.
    args: list of tuples
        Each tuple contains as the first argument the position of monopoles
        (2D array of floats), and as the second argument the corresponding charges
        (array of floats). Use efish_monopoles() to generate monopoles and
        corresponding charges.
    eps: float
        Stepsize in unit of the coordinates.
    maxiter: int
        Maximum number of iteration steps.

    Returns
    -------
    fl: 2D array of floats
        Coordinates of the computed field line.

    Example
    -------
    ```
    fig, ax = plt.subplots()
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    fl = fieldline((0, -16), [[-maxx, -maxy], [maxx, maxy]], poles1, poles2)
    plot_fieldlines(ax, [fl], 5, color='b', lw=2)
    plt.show()
    ```
    """
    bounds = np.asarray(bounds)
    p = np.array(pos0)
    n = maxiter//2
    # forward integration:
    flf = np.zeros((n, len(pos0)))
    for i in range(len(flf)):
        flf[i,:] = p
        if np.any(p < bounds[0,:]) or np.any(p > bounds[1,:]) or (bounds is not None and
                i >= 5 and np.all((flf[i,:] - flf[i-1,:])*(flf[i-1,:] - flf[i-2,:])<0)):
            flf = flf[:i,:]
            break
        uv = efield(p, *args)
        uv /= np.linalg.norm(uv)
        p = p + eps*uv
    # backward integration:
    p = np.array(pos0)
    flb = np.zeros((n, len(pos0)))
    for i in range(len(flb)):
        flb[i,:] = p
        if np.any(p < bounds[0,:]) or np.any(p > bounds[1,:]) or (bounds is not None and
                i >= 5 and np.all((flb[i,:] - flb[i-1,:])*(flb[i-1,:] - flb[i-2,:])<0)):
            flb = flb[:i,:]
            break
        uv = efield(p, *args)
        uv /= np.linalg.norm(uv)
        p = p - eps*uv
    fl = np.vstack((flb[::-2], flf[::2]))
    return fl


def squareroot_transform(values, thresh=0.0):
    """Square-root transformation keeping the sign.

    Takes the square root of positive values and takes the square root
    of the absolute values of negative values and negates the results.

    Then truncate symmetrically both positive and negative values to
    a threshold.

    The resulting transformed values give nice contour lines in a
    contour plot.

    Parameters
    ----------
    values: array of float
        The values to be transformed, i.e. potentials or field strengths.
    thresh: float or None
        Maximum absolute value of the returned values.
        Must be positive!
        If thresh equals zero, then do not apply treshold.
        If None, take the smaller of the maximum of the
        positive values or of the absolute negative values. 

    Returns
    -------
    values: array of float
        The transformed (square-rooted and thresholded) values.
    """
    values = np.array(values)
    sel = values>=0.0
    values[sel] = values[sel]**0.5
    values[np.logical_not(sel)] = -((-values[np.logical_not(sel)])**0.5)
    if thresh is None:
        thresh = min(np.max(values), -np.min(values))
    if thresh > 0:
        values[values>thresh] = thresh
        values[values<-thresh] = -thresh
    return values


def plot_fieldlines(ax, flines, pos=5, **kwargs):
    """Plot field lines with arrows.

    Parameters
    ----------
    ax: matplotlib axes
        Axes in which to plot the field lines.
    flines: list of 2D arrays
        The field lines.
    pos: float
        The position of the arrow on the field line in units of the coordinates.
    **kwargs: key word arguments
        Passed on to plot().
        Applies optional zorder argument also to arrow.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = 0.05*np.abs(xmax-xmin)
    dy = 0.05*np.abs(ymax-ymin)
    akwargs = dict()
    if 'zorder' in kwargs:
        akwargs['zorder'] = kwargs['zorder']
    lw = 1
    if 'lw' in kwargs:
        lw = kwargs['lw']
    if 'linewidth' in kwargs:
        lw = kwargs['linewidth']
    for fl in flines:
        ax.plot(fl[:,0], fl[:,1], **kwargs)
        # arrows:
        d = np.diff(fl, axis=0)
        dd = np.linalg.norm(d, axis=1)
        dist = np.cumsum(dd)
        if dist[-1] >= 6:
            idx0 = np.argmin(np.abs(dist-pos))
            if (np.abs(fl[0,0]-xmin)<dx or np.abs(fl[0,0]-xmax)<dx or
                np.abs(fl[0,1]-ymin)<dy or np.abs(fl[0,1]-ymax)<dy):
                idx0 = np.argmin(np.abs(dist[-1]-dist-pos))
            idx1 = np.argmin(np.abs(dist-0.5*dist[-1]))
            idx = min(idx0, idx1)
            adx = fl[idx+1,:] - fl[idx,:]
            ndx = np.linalg.norm(adx)
            if ndx < 1e-10:
                continue
            adx /= ndx
            posa = fl[idx+1,:] - 0.1*min(dx,dy)*adx
            posb = fl[idx+1,:]
            arrow = FancyArrowPatch(posA=posa, posB=posb, shrinkA=0, shrinkB=0,
                                    arrowstyle='fancy', mutation_scale=8*lw,
                                    connectionstyle='arc3', fill=True,
                                    color=kwargs['color'], **akwargs)
            ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots()
    maxx = 30.0
    maxy = 27.0
    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)
    fish1 = ((-8, -5), (1, 0.5), 18.0, -25)
    fish2 = ((12, 3), (0.8, 1), 20.0, 20)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = object_monopoles((-6, 0), 1.0, -0.5, poles1, poles2)
    allpoles = (poles1, poles2, poles3)
    # potential:
    pot = epotential_meshgrid(xx, yy, None, *allpoles)
    thresh = 0.65
    zz = squareroot_transform(pot/200, thresh)
    levels = np.linspace(-thresh, thresh, 16)
    ax.contourf(x, y, -zz, levels, cmap='RdYlBu')
    ax.contour(x, y, -zz, levels, zorder=1, colors='#707070',
               linewidths=0.1, linestyles='solid')
    # electric field vectors:
    n = 5
    qx, qy = np.meshgrid(x[n::2*n], y[n::2*n])
    fieldx, fieldy = efield_meshgrid(qx, qy, None, *allpoles)
    u = squareroot_transform(fieldx, 0)
    v = squareroot_transform(fieldy, 0)
    ax.quiver(qx, qy, u, v, units='xy', angles='uv', scale=2, scale_units='xy',
              width=0.07, headwidth=5)
    # field line:
    bounds = [[-maxx, -maxy], [maxx, maxy]]
    fl = fieldline((0, -16), bounds, *allpoles)
    plot_fieldlines(ax, [fl], 5, color='b', lw=2)
    plt.show()

            
if __name__ == '__main__':
    main()

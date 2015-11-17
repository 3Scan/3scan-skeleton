import numpy as np
import mayavi.mlab as mlab
import moviepy.editor as mpy

duration = 10  # duration of the animation in seconds (it will loop)

# MAKE A FIGURE WITH MAYAVI
originalIp = np.load('/Users/3scan_editing/thinning/originalIp.npy')
# fig = mlab.figure(size=(500, 500), bgcolor=(1,1,1))

# u = np.linspace(0, 2 * np.pi, 100)
contourList = [1]
l = mlab.contour3d(originalIp, contours=contourList, colormap='gray')
# xx,yy,zz = np.cos(u), np.sin(3*u), np.sin(u) # Points
# l = mlab.plot3d(xx,yy,zz, representation="wireframe", tube_sides=5,
#                 line_width=.5, tube_radius=0.2, figure=fig)

# ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF


def make_frame(t):
    """ Generates and returns the frame for time t. """
    # y = np.sin(3 * u) * (0.2 + 0.5 * np.cos(2 * np.pi * t / duration))
    # l.mlab_source.set(y=y)  # change y-coordinates of the mesh
    mlab.view(azimuth=360 * t / duration)  # camera angle
    return mlab.screenshot()  # return a RGB image

animation = mpy.VideoClip(make_frame, duration=duration)
# Video generation takes 10 seconds, GIF generation takes 25s
animation.write_videofile("microvesselsTest6.mp4", fps=20)


from mayavi import mlab
@mlab.animate
def anim():
    contourList = [1]
    f = mlab.contour3d(originalIp, contours=contourList, colormap='gray')
    while 1:
        f.scene.camera.azimuth(10)
        f.scene.render()
        yield

a = anim() # Starts the animation.


def emitCodeFromEqtn(eqtn, eqName):
    simplified = eqtn.simplify()
    uniqueIDtoSymbol = {symbol.uniqid: symbol for symbol in eqtn.inputs}

    args = ", ".join(["uint8 %s" % symbol.name for symbol in eqtn.inputs])

    # Emit the c function header
    print("uint8 %s(%s) {" % (eqName, args))
    statement = recursiveEmitter(simplified.to_ast(), uniqueIDtoSymbol)
    print("\treturn %s;" % statement)
    print("}")
    return statement


def recursiveEmitter(ast, symbolTable):
    # Input ast format is (operator, expr, expr, expr, expr. . .)
    op = ast[0]
    exprs = ast[1:]

    if op == "and":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " & ".join(subExprs) + ")"

    elif op == "or":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " | ".join(subExprs) + ")"

    elif op == "not":
        return "(! (%s))" % recursiveEmitter(exprs[0], symbolTable)

    elif op == "lit":
        # Lit can have only one operator
        symbol = exprs[0]

        if symbol < 0:
            symbol *= -1
            return "(!%s)" % symbolTable[symbol].name
        else:
            return symbolTable[symbol].name
    raise RuntimeError("No way to resolve operation named '%s' with %i arguments" % (op, len(exprs)))


def fn(*args, eqn):
	return eqn


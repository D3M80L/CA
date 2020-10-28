import numpy as np
import decimal
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def acca2dVonNeumannLocalRule(lut:np.ndarray, x:decimal, y:decimal, z:decimal, u:decimal, v:decimal) -> decimal:
    mx = 1.0 - x
    my = 1.0 - y
    mz = 1.0 - z
    mu = 1.0 - u
    mv = 1.0 - v
    
    tmp = mx * my
    result = lut[0] * tmp * mz * mu *  v
    result += lut[1] * tmp * mz *  u * mv
    result += lut[2] * tmp * mz *  u *  v
    result += lut[3] * tmp *  z * mu * mv
    result += lut[4] * tmp *  z * mu *  v
    result += lut[5] * tmp *  z *  u * mv
    result += lut[6] * tmp *  z *  u *  v

    tmp = mx * y
    result += lut[7] * tmp * mz * mu * mv
    result += lut[8] * tmp * mz * mu *  v
    result += lut[9] * tmp * mz *  u * mv
    result += lut[10] * tmp * mz *  u *  v
    result += lut[11] * tmp *  z * mu * mv
    result += lut[12] * tmp *  z * mu *  v
    result += lut[13] * tmp *  z *  u * mv
    result += lut[14] * tmp *  z *  u *  v

    tmp = x * my
    result += lut[15] * tmp * mz * mu * mv
    result += lut[16] * tmp * mz * mu *  v
    result += lut[17] * tmp * mz *  u * mv
    result += lut[18] * tmp * mz *  u *  v
    result += lut[19] * tmp *  z * mu * mv
    result += lut[20] * tmp *  z * mu *  v
    result += lut[21] * tmp *  z *  u * mv
    result += lut[22] * tmp *  z *  u *  v

    tmp = x * y
    result += lut[23] * tmp * mz * mu * mv
    result += lut[24] * tmp * mz * mu *  v
    result += lut[25] * tmp * mz *  u * mv
    result += lut[26] * tmp * mz *  u *  v
    result += lut[27] * tmp *  z * mu * mv
    result += lut[28] * tmp *  z * mu *  v
    result += lut[29] * tmp *  z *  u * mv  
    result +=           tmp *  z *  u * v
                                                                 
    if (result > 1):
        return 1

    return result

def acca1dLocalRule(lut: np.ndarray, x: decimal, y:decimal, z:decimal):
    mx = 1 - x
    my = 1 - y
    mz = 1 - z

    result = lut[0] * mx * my * mz
    result += lut[1] * mx * my *  z
    result += lut[2] * mx *  y * mz
    result += lut[3] * mx *  y *  z
    result += lut[4] *  x * my * mz
    result += lut[5] *  x * my *  z
    result += lut[6] *  x *  y * mz
    result += lut[7] *  x *  y *  z

    if (result > 1):
        return 1

    return result

def applyAcca2dVonNeumannRule(lut:np.ndarray, inputConfiguration:np.ndarray, outputConfiguration:np.ndarray, X:int, Y:int):
    for y in range(Y):
        top = (y + Y - 1) % Y * X
        bottom = (y + 1) % Y * X
        center = y * X

        for x in range(X):
            outputConfiguration[center + x] = acca2dVonNeumannLocalRule(lut, 
                inputConfiguration[top + x], # top
                inputConfiguration[center + (x + X - 1) % X], # left
                inputConfiguration[center + x], # center
                inputConfiguration[center + (x + 1) % X], # right
                inputConfiguration[bottom + x])


def applyAcca1dRule(lut:np.ndarray, inputConfiguration:np.ndarray, outputConfiguration:np.ndarray, X:int):
    for centerIndex in range(X):
        leftIndex = (X + centerIndex - 1) % X
        rightIndex = (centerIndex + 1) % X

        result = acca1dLocalRule(lut, inputConfiguration[leftIndex], inputConfiguration[centerIndex], inputConfiguration[rightIndex])

        outputConfiguration[centerIndex] = result

def iterate_1dconfiguration(I:np.ndarray, lut: np.ndarray):
    I1 = I.copy()
    I2 = I.copy()
    X = I.shape[0]

    while True:
        yield I1.copy()
        applyAcca1dRule(lut, I1, I2, X)
        tmp = I2
        I2 = I1
        I1 = tmp

def iterate_1dconfiguration_to(I:np.ndarray, lut: np.ndarray, T: int):
    iteration = iterate_1dconfiguration(I, lut)

    for _ in range(T):
        yield next(iteration)

def plot_timespatial(I: np.ndarray, ax = None):
    if (not ax):
        ax = plt.subplot(111)
    
    I = np.array(I)
    ax.imshow(I.astype(float), cmap='Blues', vmin=0, vmax=1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off', width=0)
    # draw gridlines
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

def iterate_configuration(I:np.ndarray, lut: np.ndarray):
    I1 = I.copy().flatten()
    I2 = I.copy().flatten()
    X = I.shape[1]
    Y = I.shape[0]

    while True:
        yield np.reshape(I1, I.shape)
        applyAcca2dVonNeumannRule(lut, I1, I2, X, Y)
        tmp = I2
        I2 = I1
        I1 = tmp

def iterate_configuration_to(I:np.ndarray, lut: np.ndarray, T: int):
    iteration = iterate_configuration(I, lut)

    for t in range(T):
        yield next(iteration)

def parse_1d_configuration(configuration:str):
    result = [int(x) for x in configuration.split(";")]
    return np.array(result, np.dtype(decimal.Decimal))

def parse_2d_configuration(configuration:str, x: int,  y: int):
    result = [int(x) for x in configuration.split(";")]
    return np.array(result, np.dtype(decimal.Decimal)).reshape(y,x)  

def plot_configuration(configuration: np.ndarray, ax = None):
    if (not ax):
        ax = plt.subplot(111)
        
    ax.imshow(configuration.astype(float), cmap='Blues', vmin=0, vmax=1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off', width=0)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, 3, 1))
    ax.set_yticks(np.arange(-.5, 3, 1))

def animate_iterate_configuration_to(configuration:np.ndarray, lut: np.ndarray, T: int):
    fig = plt.figure()
    ax = plt.subplot(111)
    iteration = iterate_configuration(configuration, lut)

    timespatial = [next(iteration).copy() for t in range(T)]
        
    def animate(t):
        ax.set_title(f't={t}, $\\rho$={np.mean(timespatial[t]):f}, min={np.min(timespatial[t]):f}, max={np.max(timespatial[t]):f}')
        plot_configuration(timespatial[t], ax=ax)
        plt.close()
        
    return HTML(animation.FuncAnimation(fig, animate, frames=T, interval=200, blit=False).to_jshtml())


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


# def animate_iterate_configuration_to(configuration: np.ndarray, lut: np.ndarray, T: int):
#     return HTML(plot_iterate_configuration_to(configuration, lut, T).to_jshtml())

# Ip = [int(x) for x in "1;1;1;1;0;0;1;0;0".split(";")]
# I = np.array(Ip, np.dtype(decimal.Decimal)).reshape(3,3)

# lut = [0.0694483294696466,0.0205768632252970,0.3936142824964936,0.0324256239840529,0.6413351118370271,0.3420886711599403,1.0000000000000000,0.0390656643647346,0.3323542788090380,0.5145222195645875,0.9337041925574603,0.4083933032081399,0.5558771701745073,0.3256708833263801,0.9996975188577638,0.0061853565507727,0.3477703425016097,0.0318818957581097,0.9516210370447654,0.0439863261339373,0.0098972290556301,0.8140427796400955,0.9974848794610039,0.0058144746957105,0.9227795261776912,0.9402703718388361,0.9353010574907661,0.2382834336453912,0.9878185427694743,0.9307289703828943]
# lut = np.array(lut, dtype=np.dtype(decimal.Decimal))


# for c in iterate_configuration_maxto(I, lut, 30):
#     print (c)

# I1 = np.array(Ip, dtype=np.dtype(decimal.Decimal))
# I2 = I1.copy()

# lut = [0.0694483294696466,0.0205768632252970,0.3936142824964936,0.0324256239840529,0.6413351118370271,0.3420886711599403,1.0000000000000000,0.0390656643647346,0.3323542788090380,0.5145222195645875,0.9337041925574603,0.4083933032081399,0.5558771701745073,0.3256708833263801,0.9996975188577638,0.0061853565507727,0.3477703425016097,0.0318818957581097,0.9516210370447654,0.0439863261339373,0.0098972290556301,0.8140427796400955,0.9974848794610039,0.0058144746957105,0.9227795261776912,0.9402703718388361,0.9353010574907661,0.2382834336453912,0.9878185427694743,0.9307289703828943]
# np.array(lut, dtype=np.dtype(decimal.Decimal))


#for configuration in 
# print (np.mean(I1))
# for t in range(60):
#     applyAcca2dVonNeumannRule(lut, I1, I2, 3, 3)
#     tmp = I2
#     I2 = I1
#     I1 = tmp

# minValue = min(I1)
# maxValue = max(I1)
# print (f"{minValue:.30f},{maxValue:.30f}")
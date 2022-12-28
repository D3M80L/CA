import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from decimal import *

def acca2dVonNeumannLocalRule(lut: np.ndarray, x, y, z, u, v):
    dec1 = Decimal('1')
    mx = dec1 - x
    my = dec1 - y
    mz = dec1 - z
    mu = dec1 - u
    mv = dec1 - v

    tmp = mx * my
    result = lut[0] * tmp * mz * mu * v
    result += lut[1] * tmp * mz * u * mv
    result += lut[2] * tmp * mz * u * v
    result += lut[3] * tmp * z * mu * mv
    result += lut[4] * tmp * z * mu * v
    result += lut[5] * tmp * z * u * mv
    result += lut[6] * tmp * z * u * v

    tmp = mx * y
    result += lut[7] * tmp * mz * mu * mv
    result += lut[8] * tmp * mz * mu * v
    result += lut[9] * tmp * mz * u * mv
    result += lut[10] * tmp * mz * u * v
    result += lut[11] * tmp * z * mu * mv
    result += lut[12] * tmp * z * mu * v
    result += lut[13] * tmp * z * u * mv
    result += lut[14] * tmp * z * u * v

    tmp = x * my
    result += lut[15] * tmp * mz * mu * mv
    result += lut[16] * tmp * mz * mu * v
    result += lut[17] * tmp * mz * u * mv
    result += lut[18] * tmp * mz * u * v
    result += lut[19] * tmp * z * mu * mv
    result += lut[20] * tmp * z * mu * v
    result += lut[21] * tmp * z * u * mv
    result += lut[22] * tmp * z * u * v

    tmp = x * y
    result += lut[23] * tmp * mz * mu * mv
    result += lut[24] * tmp * mz * mu * v
    result += lut[25] * tmp * mz * u * mv
    result += lut[26] * tmp * mz * u * v
    result += lut[27] * tmp * z * mu * mv
    result += lut[28] * tmp * z * mu * v
    result += lut[29] * tmp * z * u * mv
    result += tmp * z * u * v

    if (result > dec1):
        return dec1

    return result


def acca1dLocalRule(lut: np.ndarray, x, y, z):
    dec1 = Decimal('1')
    mx = dec1 - x
    my = dec1 - y
    mz = dec1 - z

    result = lut[0] * mx * my * mz
    result += lut[1] * mx * my * z
    result += lut[2] * mx * y * mz
    result += lut[3] * mx * y * z
    result += lut[4] * x * my * mz
    result += lut[5] * x * my * z
    result += lut[6] * x * y * mz
    result += lut[7] * x * y * z

    if (result > dec1):
        return dec1

    return result


def applyAcca2dVonNeumannRule(lut: np.ndarray, inputConfiguration: np.ndarray, outputConfiguration: np.ndarray, X: int, Y: int):
    for y in range(Y):
        top = (y + Y - 1) % Y * X
        bottom = (y + 1) % Y * X
        center = y * X

        for x in range(X):
            outputConfiguration[center + x] = acca2dVonNeumannLocalRule(lut,
                                                                        # top
                                                                        inputConfiguration[top + x],
                                                                        # left
                                                                        inputConfiguration[center + (
                                                                            x + X - 1) % X],
                                                                        # center
                                                                        inputConfiguration[center + x],
                                                                        # right
                                                                        inputConfiguration[center + (
                                                                            x + 1) % X],
                                                                        inputConfiguration[bottom + x])


def applyAcca1dRule(lut: np.ndarray, inputConfiguration: np.ndarray, outputConfiguration: np.ndarray, X: int):
    for centerIndex in range(X):
        leftIndex = (X + centerIndex - 1) % X
        rightIndex = (centerIndex + 1) % X

        result = acca1dLocalRule(
            lut, inputConfiguration[leftIndex], inputConfiguration[centerIndex], inputConfiguration[rightIndex])

        outputConfiguration[centerIndex] = result


def iterate_1dconfiguration(I: np.ndarray, lut: np.ndarray):
    I1 = I.copy()
    I2 = I.copy()
    X = I.shape[0]

    while True:
        yield I1.copy()
        applyAcca1dRule(lut, I1, I2, X)
        tmp = I2
        I2 = I1
        I1 = tmp


def iterate_1dconfiguration_to(I: np.ndarray, lut: np.ndarray, T: int):
    iteration = iterate_1dconfiguration(I, lut)

    for _ in range(T):
        yield next(iteration)


def plot_spacetime(I: np.ndarray, ax=None):
    if (not ax):
        ax = plt.subplot(111)

    I = np.array(I)
    ax.imshow(I.astype(float), cmap='Blues', vmin=0, vmax=1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off', width=0)
    # draw gridlines
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])


def iterate_configuration(I: np.ndarray, lut: np.ndarray):
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


def iterate_configuration_to(I: np.ndarray, lut: np.ndarray, T: int):
    iteration = iterate_configuration(I, lut)

    for t in range(T):
        yield next(iteration)


def parse_1d_configuration(configuration: str):
    c = configuration
    if (";" in configuration):
        c = c.split(";")
    result = [Decimal(x) for x in c]
    return np.array(result)


def parse_2d_configuration(configuration: str, x: int,  y: int):
    result = [Decimal(x) for x in configuration.split(";")]
    return np.array(result).reshape(y, x)


def plot_configuration(configuration: np.ndarray, ax=None):
    if (not ax):
        ax = plt.subplot(111)

    X = configuration.shape[1]
    Y = configuration.shape[0]

    ax.imshow(configuration.astype(float), cmap='Blues', vmin=0, vmax=1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off', width=0)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, X, 1))
    ax.set_yticks(np.arange(-.5, Y, 1))


def title_timespatial_default(t, configuration):
    min = np.min(configuration)
    max = np.max(configuration)
    minValue = float(f'{min:1.3f}')
    maxValue = float(f'{max:1.3f}')

    minEq = '='
    if (min > minValue):
        minEq = '>'
    elif (min < minValue):
        minEq = '<'

    maxEq = '='
    if (max > maxValue):
        maxEq = '>'
    elif (max < maxValue):
        maxEq = '<'

    maxHalf = '=0.5'
    if (max > 0.5):
        maxHalf = '>0.5'
    elif (max < 0.5):
        maxHalf = '<0.5'

    minHalf = '=0.5'
    if (min > 0.5):
        minHalf = '>0.5'
    elif (min < 0.5):
        minHalf = '<0.5'

    return f't={t}, $\\rho$={np.mean(configuration):1.3f}, min{minEq}{minValue:1.3f}({minHalf}), max{maxEq}{maxValue:1.3f}({maxHalf})'


def title_notify_when_threshold_reached(threshold):
    def notify_below(t, configuration, threshold):
        maxValue = np.max(configuration)
        minValue = np.min(configuration)

        if (maxValue <= threshold):
            return f' threshold'
        if (minValue >= (1 - threshold)):
            return f' threshold'
        return ''

    return lambda t, configuration: title_timespatial_default(t, configuration) + notify_below(t, configuration, threshold)


def set_title(ax, timespatial, t, title=None):

    if (title is None):
        title = title_timespatial_default

    result = title(t, timespatial[t])

    ax.set_title(result)


def animate_iterate_configuration_to(configuration: np.ndarray, lut: np.ndarray, T: int, title=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    iteration = iterate_configuration(configuration, lut)

    timespatial = [next(iteration).copy() for t in range(T)]

    def animate(t):
        set_title(ax, timespatial, t, title=title)
        plot_configuration(timespatial[t], ax=ax)
        plt.close()

    return HTML(animation.FuncAnimation(fig, animate, frames=T, interval=200, blit=False).to_jshtml())


def plot_polar_configuration(configuration: np.ndarray, ax=None):
    if (not ax):
        ax = plt.subplot(111, projection='polar')

    N = len(configuration)

    theta = np.linspace(0, 1, num=N+1) * 2*np.pi
    I = np.zeros(N+1)
    I[:N] = configuration[:N]
    I[N] = configuration[0]

    ax.set_rorigin(-.1)
    ax.set_rmax(1)
    ax.set_rticks([0.5, 1])
    ax.grid(True)
    p = 360 / N
    ax.set_thetagrids(np.arange(0, 360, p))
    ax.set_xticklabels([])
    ax.set_ylim([0, 1])
    ax.spines['polar'].set_visible(False)
    ax.scatter(theta, I)
    ax.plot(theta, I)

    return ax


def animate_iterate_polar_configuration_to(configuration: np.ndarray, lut: np.ndarray, T: int):
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')

    timespatial = list(iterate_1dconfiguration_to(configuration, lut, T))

    def animate(t):
        ax.cla()
        set_title(ax, timespatial, t)
        plot_polar_configuration(timespatial[t], ax=ax)
        plt.close()

    return HTML(animation.FuncAnimation(fig, animate, frames=T, interval=200, blit=False).to_jshtml())


def outer_totalistic_acca_lut(a, b, c, d) -> np.ndarray:
    a = Decimal(a)
    b = Decimal(b)
    c = Decimal(c)
    d = Decimal(d)
    dec2 = Decimal('2')
    return np.array([a, b, c, d, Decimal(b), dec2*b-a, d, dec2*d-c])


def density_conserving_2d_lut(a, b, c, d, e, f, g, h) -> np.ndarray:
    a = Decimal(a)
    b = Decimal(b)
    c = Decimal(c)
    d = Decimal(d)
    e = Decimal(e)
    f = Decimal(f)
    g = Decimal(g)
    h = Decimal(h)
    return np.array([a,
                     b,
                     c,
                     d,
                     e,
                     f,
                     -a-b+c-d+e+f,
                     g,
                     h,
                     b+g,
                     -a+c+h,
                     b+2*d-f+g,
                     -a+b+d+e-f+h,
                     b+d+g,
                     -2*a+c+e+h,
                     1-a-b-d-g,
                     1-b - d-g,
                     1-d - h,
                     1-b + c - d - h,
                     1-b + d - e - g,
                     1-b - g,
                     1+a - b - e + f - h,
                     1-2*b + c - d + f - h,
                     1-c - d,
                     1-c - d - g + h,
                     1+a + b - c - d - h + g,
                     1-d,
                     1+a + b - c + 2*d - e - f,
                     1+b - c + d - f - g + h,
                     1+2*a + b - c + d - e + g - h])

def density_conserving_rotation_symmetric_2d_lut(alpha:float) -> np.ndarray:
    return np.array([
        alpha,
        alpha,
        2*alpha,
        1-4*alpha,
        1-3*alpha,
        1-3*alpha,
        1-2*alpha,
        alpha,
        2*alpha,
        2*alpha,
        3*alpha,
        1-3*alpha,
        1-2*alpha,
        1-2*alpha,
        1-alpha,
        1-4*alpha,
        2*alpha,
        2*alpha,
        3*alpha,
        1-3*alpha,
        1-2*alpha,
        1-2*alpha,
        1-alpha,
        2*alpha,
        3*alpha,
        3*alpha,
        4*alpha,
        1-2*alpha,
        1-alpha,
        1-alpha
    ])

def diffusion_2d_lut() -> np.ndarray:
    dec5 = Decimal('5')
    dec1 = Decimal('1')
    dec2 = Decimal('2')
    return density_conserving_2d_lut(a=dec1/dec5, b=dec1/dec5, c=dec2/dec5, d=dec1/dec5, e=dec2/dec5, f=dec2/dec5, g=dec1/dec5, h=dec2/dec5)


def diffusion_1d_lut() -> np.ndarray:
    dec0 = Decimal('0')
    dec1 = Decimal('1')
    dec2 = Decimal('2')
    dec3 = Decimal('3')
    return np.array([dec0, dec1/dec3, dec1/dec3, dec2/dec3, dec1/dec3, dec2/dec3, dec2/dec3, dec1])


def density_conserving_rotation_symmetric_2d_lut(alpha) -> np.ndarray:
    if (alpha < 0 or alpha > 0.25):
        raise ValueError('The alpha value can be only in range [0, 0.25].')

    return density_conserving_2d_lut(a=alpha, b=alpha, c=2*alpha, d=1-4*alpha, e=1-3*alpha, f=1-3*alpha, g=alpha, h=2*alpha)

def density_conserving_1d_lut(a, b, c) -> np.ndarray:
    a = Decimal(a)
    b = Decimal(b)
    c = Decimal(c)
    dec0 = Decimal('0')
    dec1 = Decimal('1')
    return np.array([
        dec0,
        a,
        c,
        dec1-b,
        dec1-a-c,
        dec1-c,
        b+c,
        dec1])

def truncate(x):
    truncated = Decimal(int(x * 100000)) / Decimal(100000)
    dots = ""
    if (truncated != x):
        dots = "\ldots"
    return f"{truncated}{dots}"

def display_lut(lut: np.ndarray):
    if (len(lut) == 6 or len(lut) == 30):
        lut = np.concatenate(([0], lut, [1]))
    
    if (len(lut) != 8 and len(lut) != 32):
        print (len(lut))
        raise ValueError('Only 1D or 2D LUT is allowed.')

    result = "<table style='border: 1px solid black;border-collapse: collapse;width:100%'><caption>LUT representation</caption>"
    for y in range(len(lut) // 8):
        header = ''.join([f'<td style="text-align:center;border: 1px solid black">$l_{{{x+8*y}}}$</td>' for x in range(8)])
        values = ''.join([f'<td style="text-align:center;border: 1px solid black">${truncate(x)}$</td>' for x in lut[y*8:y*8+8]])
        result += f'<tr>{header}</tr><tr>{values}</tr>'
    result += "</table>"

    return HTML(result)

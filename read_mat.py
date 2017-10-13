import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# import Image
import matplotlib.widgets as widgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# path = '../spectrum_correction/data/0920_01_3'
# data = sio.loadmat(path)
# sio.savemat('light3.mat',{'light3':data['0920_01_3']})
#
# path = '../spectrum_correction/data/0920_01_2'
# data = sio.loadmat(path)
# sio.savemat('light2.mat',{'light2':data['0920_01_2']})
#
# path = '../spectrum_correction/data/0920_01_1'
# data = sio.loadmat(path)
# sio.savemat('light1.mat',{'light1':data['0920_01_1']})

light1 = sio.loadmat('light1')
# light2 = sio.loadmat('light2')
# light3 = sio.loadmat('light3')

def onselect(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax2.set_ylim(erelease.ydata,eclick.ydata)
    ax2.set_xlim(eclick.xdata,erelease.xdata)
    fig2.canvas.draw()

arr = np.asarray(light1['light1'][:,:,1])

# fig= Figure(1)
# canvas = FigureCanvas(fig)
# ax1 = fig.add_axes()

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plt.imshow(arr,cmap='gray')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
plt_image=plt.imshow(arr,cmap='gray')

rs=widgets.RectangleSelector(
    ax1, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))

fig3 = plt.figure(3)
ax=fig3.add_subplot(111)

plt.plot()

plt.show()
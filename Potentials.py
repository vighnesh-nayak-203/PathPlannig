import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from Opt import *

x_init=np.array([1,1]).reshape(2,1)
x_final=np.array([13,13]).reshape(2,1)


obstacles=np.array([[3,3],[6,7],[11,14],[3,9],[10,10]]).reshape(5,2)
obstacles=obstacles.astype('float64')

def func(x):
    x=np.array(x).reshape(2,1)
    p=norm(x-x_final)
    
    if x[0]<=0.5:
        p+=400*abs(0.5-x[0])**2
    if x[0]>=14.5:
        p+=400*abs(x[0]-14.5)**2
    if x[1]<=0.5:
        p+=400*abs(0.5-x[1])**2
    if x[1]>=14.5:
        p+=400*abs(x[1]-14.5)**2

    for i in obstacles:
        i=i.reshape(2,1)
        r=1
        d0=r+1
        dq=np.sqrt(norm(x-i))
        if dq<=r:
            p+=100
        elif dq<=d0:
            p+=400*((1/(dq))-(1/(d0)))**2
    return p

def plot():
    def p(x,y):
        return func([x,y])
    f=np.vectorize(p)
    X,Y=np.meshgrid(np.linspace(0,15,100),np.linspace(0,15,100))
    Z=f(X,Y)

    fig,ax=plt.subplots(1,1,figsize=(10,10),subplot_kw={'projection':'3d'})
    surf=ax.plot_surface(X,Y,Z, linewidth=0.2, antialiased=False,cmap=mpl.colormaps['plasma'])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_title(r'Plot 2: Surface plot of $f(x_1,x_2)$',fontweight='bold',fontsize=14)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

path,iter=FRCG(func,x_init,x_final)
print(f'Path Length: {np.sum([np.sqrt(norm(i)) for i in np.diff(path,axis=0)])}')
fig=plt.figure(figsize=(10,10))
plt.xlim(0,15)
plt.ylim(0,15)
plt.plot(path[:,0],path[:,1],color='k',marker='.',label='Path')

for i in obstacles:
    c=plt.Circle(i,1,color='r',label='Obstacle')
    plt.gca().add_patch(c)
plt.scatter([x_init[0],x_final[0]],[x_init[1],x_final[1]],color='g',label='Start/End')
plt.legend()
plt.show()
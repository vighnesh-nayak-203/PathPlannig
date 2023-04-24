""" 
This file contains optimization functions for finding path.
"""

import numpy as np

def gradient(func, x_input):
  n=2
  h=0.001
  x_input = np.array(x_input).reshape(n, 1)
  delF = np.zeros((n, 1))

  for i in range(n):
    e = np.zeros((n, 1))
    e[i] = 1

    delF[i] = ((func(x_input+h*e)-func(x_input-h*e))/(2*h))

  return delF

def hessian(func, x_input):
  h=0.001
  n=2

  x_input = np.array(x_input).reshape(n, 1)
  del2F = np.zeros((n, n))

  for i in range(n):
    ei = np.zeros((n, 1))
    ei[i] = 1

    del2F[i][i] = (func(x_input+h*ei)+func(x_input-h*ei)-2*func(x_input))/(h**2)

    for j in range(i+1, n):
        ej = np.zeros((n, 1))
        ej[j] = 1

        A = func(x_input+h*ei+h*ej)
        B = func(x_input-h*ei-h*ej)
        C = func(x_input+h*ei-h*ej)
        D = func(x_input-h*ei+h*ej)

        del2F[i][j] = (A+B-C-D)/(4*h**2)
        del2F[j][i] = del2F[i][j]
  
  return del2F

def backTrack(func,xk,pk):
    a_=5
    c=0.1
    rho=0.8
    fk=func(xk)
    G=np.matmul(gradient(func,xk).T,pk)[0][0]
    while (func(xk+a_*pk)>fk+c*a_*G):
        a_*=rho
    return a_

def norm(vect):
    return abs(np.matmul(vect.T,vect))
           
def steepest_descent(func, x_initial, x_final):
    N=15000
    n=2
    h=10**-6

    num_iterations=1
    x_iterations=np.zeros((N,n))
    f_values=np.zeros((N,1))

    x_iterations[0]=x_initial[:].reshape(n,)
    f_values[0]=func(x_iterations[0])
    
    for num_iterations in range(1,N):
        xk=x_iterations[num_iterations-1].reshape(n,1)
        delF = gradient(func,xk)

        if norm(delF)<h:
            num_iterations-=1
            break

        if norm(xk-x_final)<10**-2:
            num_iterations-=1
            break

        pk=-1*delF
        ak=0.02
        pk=pk/np.sqrt(norm(pk))

        x_iterations[num_iterations]=(xk+ak*pk).reshape(n,)
        f_values[num_iterations]=func(x_iterations[num_iterations])
    

    x_iterations=x_iterations[:num_iterations+1]
    f_values=f_values[:num_iterations+1]

    
    return x_iterations,num_iterations
    
def newton_method(func, x_initial, x_final):
    N=15000
    n=2
    h=10**-6

    num_iterations=1
    x_iterations=np.zeros((N,n))
    f_values=np.zeros((N,1))

    x_iterations[0]=x_initial[:].reshape(n,)
    f_values[0]=func(x_iterations[0])
    
    for num_iterations in range(1,N):
        xk=x_iterations[num_iterations-1].reshape(n,1)
        delF = gradient(func,xk)
        del2F = hessian(func,xk)

        if norm(xk-x_final)<10**-2 or norm(delF)<h: 
            num_iterations-=1
            break

        try:
            p=-1*np.linalg.inv(del2F)
        except:
            num_iterations-=1
            break

        pk=np.matmul(p,delF)
        ak=0.02
        pk=pk/np.sqrt(norm(pk))

        x_iterations[num_iterations]=(xk+ak*pk).reshape(n,)
        f_values[num_iterations]=func(x_iterations[num_iterations])


    x_iterations=x_iterations[:num_iterations+1]
    f_values=f_values[:num_iterations+1]

    return x_iterations,num_iterations

def quasi_newton_method(func, x_initial, x_final):
    N=15000
    n=2
    h=10**-6

    num_iterations=1
    x_iterations=np.zeros((N,n))
    f_values=np.zeros((N,1))

    x_iterations[0]=x_initial[:].reshape(n,)
    f_values[0]=func(x_iterations[0])
    
    Ck=np.identity(n)
    for num_iterations in range(1,N):
        xk=x_iterations[num_iterations-1].reshape(n,1)
        delF = gradient(func,xk)

        if norm(delF)<h:
            num_iterations-=1
            break

        if norm(xk-x_final)<10**-2:
            num_iterations-=1
            break

        pk=-1*np.matmul(Ck,delF)
        ak=0.02
        pk=pk/np.sqrt(norm(pk))

        x_iterations[num_iterations]=(xk+ak*pk).reshape(n,)
        f_values[num_iterations]=func(x_iterations[num_iterations])

        s=x_iterations[num_iterations].reshape(n,1)-xk
        y=gradient(func,x_iterations[num_iterations])-gradient(func,xk)
        c=np.matmul(y.T,s)

        I=np.identity(n)
        A=np.matmul((I-np.matmul(s,y.T)/c),Ck)
        Ck=np.matmul(A,(I-np.matmul(y,s.T)/c))+np.matmul(s,s.T)/c
    
    
    x_iterations=x_iterations[:num_iterations+1]
    f_values=f_values[:num_iterations+1]

    return x_iterations,num_iterations

def FRCG(func, x_initial,x_final):
    np.seterr(invalid='ignore')
    N=15000
    n=2
    h=10**-6

    num_iterations=1
    x_iterations=np.zeros((N,n))
    f_values=np.zeros((N,1))

    x_iterations[0]=x_initial[:].reshape(n,)
    f_values[0]=func(x_iterations[0])
    pk=-1*gradient(func,x_iterations[0])
    
    for num_iterations in range(1,N):
        xk=x_iterations[num_iterations-1].reshape(n,1)
        delF = gradient(func,xk)

        if norm(xk-x_final)<10**-2 or norm(delF)<h:
            num_iterations-=1
            break

        ak=0.02
        xk_n=xk+ak*pk

        x_iterations[num_iterations]=(xk_n).reshape(n,)
        f_values[num_iterations]=func(xk_n)

        bk=norm(gradient(func,xk_n))/norm(gradient(func,xk))
        if num_iterations%1000==0 or bk==np.nan or bk==np.inf:
            print(bk)
            bk=0
        pk=-1*gradient(func,xk_n)+bk*pk
        pk=pk/np.sqrt(norm(pk))

   
    x_iterations=x_iterations[:num_iterations+1]
    f_values=f_values[:num_iterations+1]
    
    return x_iterations,num_iterations

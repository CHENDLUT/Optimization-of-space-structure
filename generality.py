from math import gcd, ceil
import itertools
from scipy import sparse
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from mpl_toolkits.mplot3d import Axes3D
#define Coefficent matrix B
def cscB(Nd,Cn,dp):
    m,n1,n2 = len(Cn),Cn[:,0].astype(int),Cn[:,1].astype(int)
    l,X,Y,Z = Cn[:,2],Nd[n2,0]-Nd[n1,0],Nd[n2,1]-Nd[n1,1],Nd[n2,2]-Nd[n1,2]
    d0,d1,d2,d3,d4,d5 = dp[n1*3], dp[n1*3+1], dp[n1*3+2], dp[n2*3], dp[n2*3+1], dp[n2*3+2]
    s = np.concatenate((-X/l*d0,-Y/l*d1,-Z/l*d2,X/l*d3,Y/l*d4,Z/l*d5))
    r = np.concatenate((n1*3,n1*3+1,n1*3+2,n2*3,n2*3+1,n2*3+2))
    c = np.concatenate((np.arange(m),np.arange(m),np.arange(m),np.arange(m),np.arange(m),np.arange(m)))
    return sparse.coo_matrix((s,(r,c)),shape = (len(Nd)*3, m))
#define Solving funcion of linear programming
def SolveLP(Nd, Cn, Fi, dp, st, sc, am):
    l = [col[2] + am for col in Cn]
    B = cscB(Nd, Cn, dp)
    a = cvx.Variable(len(Cn))
    obj = cvx.Minimize(np.transpose(l) * a)
    q, eqn, cons = [], [], [a>=0]
    for k, fk in enumerate(Fi):
        q.append(cvx.Variable(len(Cn)))
        eqn.append(B * q[k] == fk * dp)
        cons.extend([eqn[k], q[k] >= -sc * a,q[k] <= st * a])
    prob = cvx.Problem(obj, cons)
    vol = prob.solve()
    q = [np.array(qi.value).flatten() for qi in q]
    a = np.array(a.value).flatten()
    u = [-np.array(eqnk.dual_value).flatten() for eqnk in eqn]
    return vol, a, q, u, l
#define stopViolation
def stopViolation(Nd, MP, dp, st, sc, u, am):
    lst = np.where(MP[:,3]==False)[0]
    Cn = MP[lst]
    l = Cn[:,2] + am
    B = cscB(Nd, Cn, dp).tocsc()
    y = np.zeros(len(Cn))
    for uk in u:
        yk = np.multiply(B.transpose().dot(uk)/l, np.array([[st], [-sc]]))
        y += np.amax(yk, axis=0)
    vioCn = np.where(y>1.0001)[0]
    vioSort = np.flipud(np.argsort(y[vioCn]))
    num = ceil(min(len(vioSort), 0.05*max( [len(Cn)*0.05, len(vioSort)])))
    for i in range(num):
        MP[lst[vioCn[vioSort[i]]]][3] = True
    return num == 0
#define Trussvisualization
def Trussvisualization(Nd, Cn, a, q, threshold, str, update = True):
    plt.ion() if update else plt.ioff()
    plt.clf(); plt.axis('off'); plt.draw()
    ax1 = plt.axes(projection='3d')
    plt.title(str)
    tk = 3/max(a)
    for i in [i for i in range(len(a)) if a[i] >= threshold]:
        if all([q[lc][i] >= 0 for lc in range(len(q))]): c = 'r'
        elif all ([q[lc][i] <= 0 for lc in range(len(q))]): c='b'
        else: c = 'tab:gray'
        pos = Nd[Cn[i, [0, 1]].astype(int), :]
        plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], c, linewidth = a[i] * tk)
    plt.pause(0.01) if update else plt.show()
#define main function
def trussopt(length,width,height,st,sc,am):
    xv, yv, zv = np.meshgrid(range(length + 1), range(width + 1), range(height + 1))
    pts = [Point(xv.flat[i], yv.flat[i], zv.flat[i]) for i in range(xv.size)]
    polylist = [[(i, 0, 0),(i, width, 0),(i,width,height),(i, 0, height)] for i in range(length+1)]
    Nd=[[0,0,0]]
    for pl in polylist:
        poly = Polygon(pl)
        Nd1 = np.array([[pt.x, pt.y, pt.z] for pt in pts if poly.intersects(pt)])
        Nd = np.vstack((Nd,Nd1))
    Nd = np.delete(Nd,0,axis = 0)
    dp, Fi, MP = np.ones((len(Nd), 3)), [], []
    for i, nd in enumerate(Nd):
        Fi += [0, 0, -1] if (nd[0] == length/2 and nd[1] == width/2 and nd[2]==0) else [0, 0, 0]
        if (nd[0] == 0 and nd[1]==0 and nd[2]==0): dp[i, :] = [0, 0, 0]
        if (nd[0] == 0 and nd[1]==width and nd[2]==0): dp[i, :] = [0, 1, 0]
        if (nd[0] == length and nd[1]==0 and nd[2]==0): dp[i, :] = [1, 1, 0]
        if (nd[0] == length and nd[1]==width and nd[2]==0): dp[i, :] = [1, 1, 0]
    for i, j in itertools.combinations(range(len(Nd)), 2):
        dx, dy, dz = abs(Nd[i][0] - Nd[j][0]), abs(Nd[i][1] - Nd[j][1]), abs(Nd[i][2] - Nd[j][2])
        if gcd(int(dx), int(dy), int(dz)) == 1 or am != 0:
            MP.append( [i, j, np.sqrt(dx**2 + dy**2 + dz**2), False] )
    MP, dp = np.array(MP), np.array(dp).flatten()
    Fi = [Fi[i:i + len(Nd) * 3] for i in range(0, len(Fi), len(Nd) * 3)]
    print('Nodes: %d Members: %d' % (len(Nd), len(MP)))
    for pm in [p for p in MP if p[2] <= 1.75]:
        pm[3] = True
    for itr in range(1, 100):
        Cn = MP[MP[:, 3] == True]
        vol, a, q, u, l = SolveLP(Nd, Cn, Fi, dp, st, sc, am)
        print("Itr: %d,vol: %f,mems:%d" % (itr, vol, len(Cn)))
        Trussvisualization(Nd, Cn, a, q, max(a) * 1e-3, "Itr:" + str(itr))
        xxx, abc = 0,0
        for i in range(len(q)):
            xxx=xxx+q[i]*l[i]
            for x_x,xx in enumerate(xxx):
                xx = xx if xx>0 else -1*xx
                abc = abc+xx
            print( 4.354 * abc )
        if stopViolation(Nd, MP, dp, st, sc, u, am): break
    print("Volume: %f" % (vol))
    Trussvisualization(Nd, Cn, a, q, max(a) * 1e-3, "Finished", False)
if __name__ == '__main__':
    trussopt(length = 200, width = 100, height = 10, st = 1, sc = 1, am = 0)

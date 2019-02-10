import numpy as np


def sse(centers, points, cluster):
    return np.sum([np.linalg.norm(p - centers[cluster[i]])**2 for i, p in enumerate(points)])


def smartinit(k, points):
    def d(point, centers):
        return np.min([np.linalg.norm(point - c)**2 for c in centers])

    def distrib(centers, points):
        if centers == []:
            return 1/len(points)*np.ones(len(points))
        else:
            L = np.array([d(p, centers) for p in points])
            return 1/np.sum(L)*L
    c = []
    for i in range(k):
        indx = np.random.choice(len(points), 1, p = distrib(c, points))[0]
        #print(indx)
        c.append(points[indx])
        points = np.concatenate([points[:indx], points[indx+1:]])
    return c


def kmeans(k, points, init='uniform'):
    if init=='plusplus':
        centers = smartinit(k, points[:, :])
    else:
        centers = [points[i] for i in np.random.choice(len(points), k, replace=False)]
    
    cluster = [0 for i in range(len(points))]
    
    no_convergence = True
    while no_convergence:
        no_convergence = False
        new_centers = [[0,0] for i in range(k)]
        for i, p in enumerate(points):
            updt = np.argmin([np.linalg.norm(p - c) for c in centers])
            if(updt != cluster[i]):
                no_convergence = True
                cluster[i] = updt
            new_centers[cluster[i]][0] += p
            new_centers[cluster[i]][1] += 1
        centers = [s/n for s,n in new_centers]
    
    return np.array(cluster), centers, sse(centers, points, cluster)
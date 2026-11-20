import math
EPS=1e-8

def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def on_segment(a,b,p):
    return abs(cross(a,b,p))<=EPS and min(a[0],b[0])-EPS<=p[0]<=max(a[0],b[0])+EPS and min(a[1],b[1])-EPS<=p[1]<=max(a[1],b[1])+EPS

def intersect(a,b,c,d):
    if any((on_segment(a,b,p) for p in (c,d))) or any((on_segment(c,d,p) for p in (a,b))): return True
    return cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0

def validate_polygon(p):
    if len(p)<3 or any(len(v)!=2 or not all(math.isfinite(float(x)) for x in v) for v in p): raise ValueError('invalid polygon')
    if len(set(tuple(v) for v in p))!=len(p): raise ValueError('duplicate vertices; omit closing vertex')
    n=len(p)
    if abs(sum(p[i][0]*p[(i+1)%n][1]-p[(i+1)%n][0]*p[i][1] for i in range(n)))<EPS: raise ValueError('zero area')
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1): continue
            if intersect(p[i],p[(i+1)%n],p[j],p[(j+1)%n]): raise ValueError('self intersection')
    return p

def inside(pt,p):
    if any(on_segment(p[i],p[(i+1)%len(p)],pt) for i in range(len(p))): return True
    x,y=pt; result=False
    for i,a in enumerate(p):
        b=p[(i+1)%len(p)]
        if (a[1]>y)!=(b[1]>y) and x<a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]): result=not result
    return result

def outside(box,p):
    x1,y1,x2,y2=box
    q=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    if any(inside(v,p) for v in q): return False
    if any(x1<=x<=x2 and y1<=y<=y2 for x,y in p): return False
    return not any(intersect(q[i],q[(i+1)%4],p[j],p[(j+1)%len(p)]) for i in range(4) for j in range(len(p)))

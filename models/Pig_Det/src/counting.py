from collections import deque

class CountSession:
    """One sample per integer video second; gaps reset stability windows."""
    def __init__(self,registered,duration=900,recount=120,stable=5,min_coverage=.8):
        if type(registered)!=int or registered<1 or min(duration,recount)<stable or stable<1 or not 0<min_coverage<=1: raise ValueError('invalid count config')
        self.registered=registered; self.duration=duration; self.recount=recount
        self.stable=stable; self.min_coverage=min_coverage
        self.phase='initial'; self.start=None; self.last=None; self.window=deque(maxlen=stable)
        self.best=None; self.image=None; self.samples=0; self.first=None; self.first_image=None
        self.outcome=None; self.phase_history=[]

    def sample(self,t,count,image):
        if type(t)!=int or type(count)!=int or count<0: raise ValueError('integer second/count required')
        if self.phase=='done': return self.outcome
        if self.start is None: self.start=t
        if self.last is not None and t<=self.last: raise ValueError('nonmonotonic samples')
        if self.last is not None and t!=self.last+1: self.window.clear()
        self.last=t; self.samples+=1; self.window.append((count,image))
        if len(self.window)==self.stable:
            value=min(n for n,_ in self.window)
            if self.best is None or value>self.best:
                self.best=value; self.image=max(self.window,key=lambda v:v[0])[1]
        length=self.duration if self.phase=='initial' else self.recount
        if t-self.start+1<length: return None
        coverage=min(1.,self.samples/(t-self.start+1))
        self.phase_history.append({'phase':self.phase,'estimate':self.best,'coverage':coverage})
        if self.best is None or coverage<self.min_coverage:
            self.phase='done'; self.outcome={'status':'insufficient_data','coverage':coverage}; return self.outcome
        if self.phase=='initial' and self.best<self.registered:
            self.first=self.best; self.first_image=self.image
            self.phase='recount'; self.start=None; self.last=None; self.window.clear(); self.best=None; self.image=None; self.samples=0
            return None
        n=self.best
        if self.first is not None and self.first>n: n=self.first; self.image=self.first_image
        status='dem_du' if n==self.registered else ('dem_thieu' if n<self.registered else 'overcount_review')
        self.phase='done'; self.outcome={'status':status,'count':n,'registered':self.registered,'image':self.image,'coverage':coverage}
        return self.outcome

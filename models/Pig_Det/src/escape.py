class EscapeDetector:
    def __init__(self,hold=5,cooldown=300,max_gap=1,cooldown_scope='pen'):
        if hold<=0 or cooldown<0 or max_gap<=0 or cooldown_scope not in ('pen','track'): raise ValueError('invalid escape config')
        self.hold=hold; self.cooldown=cooldown; self.gap=max_gap; self.scope=cooldown_scope
        self.tracks={}; self.alerts={}; self.last={}

    def update(self,pen,t,outside_ids):
        if pen in self.last and t<=self.last[pen]: raise ValueError('nonmonotonic time')
        self.last[pen]=t; ids=set(outside_ids)
        for key in list(self.tracks):
            if key[0]==pen and key[1] not in ids: del self.tracks[key]
        ready=[]
        for ident in sorted(ids):
            key=(pen,ident); start,last=self.tracks.get(key,(t,t))
            if t-last>self.gap: start=t
            self.tracks[key]=(start,t)
            alert_key=pen if self.scope=='pen' else key
            if t-start>=self.hold and t-self.alerts.get(alert_key,float('-inf'))>=self.cooldown:
                ready.append(ident)
        if ready:
            for ident in ready: self.alerts[pen if self.scope=='pen' else (pen,ident)]=t
        return ready

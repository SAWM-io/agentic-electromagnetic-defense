"""Paced public HTTP calls. No automatic retries or model-provider access."""
import time,threading
DISPATCH_SPACING_S=2.
POST_CALL_WAIT_S=2.

class DispatchPacer:
    """Short metadata locks; waits and actual calls stay outside the lock."""
    def __init__(self,*,clock=time.monotonic,sleep=time.sleep):
        self.clock=clock;self.sleep=sleep;self.lock=threading.Lock();self.next_start=0.;self.events=[]
    def call(self,operation,function,*args,**kwargs):
        while True:
            with self.lock:
                now=self.clock();delay=max(0.,self.next_start-now)
                if delay<=0:
                    self.next_start=now+DISPATCH_SPACING_S;started=now;break
            self.sleep(delay)
        accepted=False
        try:
            result=function(*args,**kwargs);accepted=True;return result
        finally:
            ended=self.clock()
            with self.lock:
                self.events.append({'operation':operation,'started_monotonic':started,'transport_end_monotonic':ended,
                                    'returned_successfully':accepted,'post_call_wait_s':POST_CALL_WAIT_S})
            self.sleep(POST_CALL_WAIT_S)


class PacedClient:
    def __init__(self,client,pacer):self.client=client;self.pacer=pacer
    def __getattr__(self,name):return getattr(self.client,name)
    def recall_models(self,*args,**kwargs):return self.pacer.call('recall',self.client.recall_models,*args,**kwargs)
    def observe(self,*args,**kwargs):return self.pacer.call('observe',self.client.observe,*args,**kwargs)
    def remember_model(self,*args,**kwargs):return self.pacer.call('remember',self.client.remember_model,*args,**kwargs)
    def stimulate(self,*args,**kwargs):return self.pacer.call('stimulate',self.client.stimulate,*args,**kwargs)
    def flashback_models(self,*args,**kwargs):return self.pacer.call('flashback',self.client.flashback_models,*args,**kwargs)
    def reconnect(self):return PacedClient(self.client.reconnect(),self.pacer)
    def close(self):return self.client.close()

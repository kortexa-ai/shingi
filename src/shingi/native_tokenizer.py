"""Use the pinned native vocabulary without allocating model weights or a GPU."""
import json
import os
import subprocess
from .backend import NativeReadout


class NativeTokenizer:
    def __init__(self, executable, model, context=16384):
        env={**os.environ,'SHINGI_VOCAB_ONLY':'1'}
        self.process=subprocess.Popen([str(executable),str(model),str(context)],env=env,
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        try:
            ready=self._read(60)
            if not ready.get('ready'):raise RuntimeError('native tokenizer failed')
        except BaseException:
            self.close();raise
    _read=NativeReadout._read
    close=NativeReadout.close
    def encode(self,prompt,labels):
        self.process.stdin.write(json.dumps({'prompt':prompt,'labels':labels,'tokenize_only':True})+'\n')
        self.process.stdin.flush()
        return self._read(30)

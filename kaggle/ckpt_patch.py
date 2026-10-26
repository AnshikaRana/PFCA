a5="        D = _cached(key, lambda: self.encode(self._doc_texts(task, ids), False, ids, task))\n"
assert s.count(a5)==1,"ckpt"
s=s.replace(a5,a5.replace("self.encode(self._doc_texts(task, ids), False, ids, task)","_chunked(self, key, self._doc_texts(task, ids), ids, task)"))
s=s.replace("\ndef topk_dense(",'''
def _chunked(r,key,T,ids,task,n=4000):
    import glob
    ck="/kaggle/working/ckpt";os.makedirs(ck,exist_ok=True);out=[]
    for c,i in enumerate(range(0,len(T),n)):
        p=f"{ck}/{key}_{c:04d}.npy";q=glob.glob(f"/kaggle/input/**/{key}_{c:04d}.npy",recursive=True)
        if not os.path.exists(p) and q:np.save(p,np.load(q[0]))
        if os.path.exists(p):a=np.load(p)
        else:a=r.encode(T[i:i+n],False,ids[i:i+n],task);np.save(p,a);print("ckpt",c,i+len(a),len(T),flush=True)
        out.append(a)
    return np.concatenate(out)

def topk_dense(''',1)
print("ckpt patch on",flush=True)

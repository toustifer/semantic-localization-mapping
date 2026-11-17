"""Read an official ZIP with verified HTTP byte ranges; never fetch the 79GiB ZIP."""
import io
import json
import time
import zipfile
import hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent / 'data'
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'

class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        h = self.session.head(url, timeout=45)
        h.raise_for_status()
        self.size = int(h.headers['Content-Length'])
        self.pos = 0
        self.cache = {}
        self.downloaded = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size-self.pos
        n = min(n, self.size-self.pos)
        if n <= 0: return b''
        # Cache aligned blocks, retaining only two to bound memory.
        block = 4*1024*1024
        out = bytearray()
        while n:
            start = self.pos//block*block
            end = min(start+block, self.size)-1
            if start not in self.cache:
                for attempt in range(4):
                    try:
                        r = self.session.get(self.url, headers={'Range':f'bytes={start}-{end}'},timeout=(20,90),stream=True)
                        with r:
                            if r.status_code != 206: raise RuntimeError(f'Range unsupported: {r.status_code}')
                            expected=f'bytes {start}-{end}/{self.size}'
                            if r.headers.get('Content-Range') != expected: raise RuntimeError('Wrong content range')
                            data=r.content
                            if len(data)!=end-start+1: raise RuntimeError('Truncated range')
                        break
                    except Exception:
                        if attempt==3: raise
                        time.sleep(2)
                if len(self.cache)>=2: self.cache.pop(next(iter(self.cache)))
                self.cache[start]=data
                self.downloaded+=len(data)
            offset=self.pos-start
            count=min(n,len(self.cache[start])-offset)
            out.extend(self.cache[start][offset:offset+count])
            self.pos+=count
            n-=count
        return bytes(out)

def extract_selected(filename, predicate):
    remote=RemoteZip(BASE+filename)
    records=[]
    with zipfile.ZipFile(remote) as archive:
        entries=[z for z in archive.infolist() if not z.is_dir() and predicate(z.filename)]
        print(filename, 'selected',len(entries),'uncompressed_MiB',round(sum(z.file_size for z in entries)/2**20,1),flush=True)
        if not entries: raise RuntimeError('No matching entries')
        for i,z in enumerate(entries):
            target=(ROOT/z.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()): raise RuntimeError('Unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size==z.file_size:
                import zlib
                content=target.read_bytes()
                if zlib.crc32(content)&0xffffffff != z.CRC: content=archive.read(z)
            else: content=archive.read(z)
            if '/velodyne/' in z.filename and len(content)%16: raise RuntimeError('Invalid XYZI size')
            temp=target.with_suffix(target.suffix+'.part')
            temp.write_bytes(content)
            temp.replace(target)
            records.append({'path':z.filename,'bytes':len(content),'crc32':z.CRC,'sha256':hashlib.sha256(content).hexdigest()})
            if (i+1)%20==0 or i==len(entries)-1: print('extracted',i+1,'/',len(entries),'transferred_MiB',round(remote.downloaded/2**20,1),flush=True)
    return {'url':BASE+filename,'remote_zip_bytes':remote.size,'downloaded_bytes':remote.downloaded,'files':records}

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    report=[]
    selections=[('data_odometry_calib.zip',lambda p:'/sequences/06/' in p),
                ('data_odometry_poses.zip',lambda p:p.endswith('/06.txt')),
                ('data_odometry_velodyne.zip',lambda p:'/sequences/06/velodyne/' in p and p.endswith('.bin') and int(Path(p).stem)<200)]
    for name,select in selections:
        report.append(extract_selected(name,select))
        (ROOT/'download_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE',str(ROOT),flush=True)

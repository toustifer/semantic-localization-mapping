import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent/'data'
seq=root/'dataset/sequences/06'
files=sorted((seq/'velodyne').glob('*.bin'))
assert [f.name for f in files]==[f'{i:06}.bin' for i in range(200)]
times=np.loadtxt(seq/'times.txt')
poses=np.loadtxt(root/'dataset/poses/06.txt')
assert len(times)>=200 and poses.shape[0]>=200 and poses.shape[1]==12
assert np.isfinite(poses).all() and np.all(np.diff(times)>0)
counts=[]
for f in files:
    points=np.fromfile(f,dtype='<f4').reshape(-1,4)
    assert len(points)>0 and np.isfinite(points).all(),f.name
    counts.append(len(points))
for source in json.loads((root/'download_manifest.json').read_text()):
    for entry in source['files']:
        assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest()==entry['sha256']
result={'sequence':'06','first_frame':0,'last_frame':199,'frames':200,
        'duration_seconds':float(times[199]-times[0]),'points_min':min(counts),'points_max':max(counts),
        'pointcloud_bytes':sum(f.stat().st_size for f in files),'finite':True,'hashes_verified':True,
        'pose_note':'Official odometry camera poses. Convert with calib Tr before LiDAR frame evaluation.',
        'labels':'No SemanticKITTI manual labels downloaded.'}
(root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

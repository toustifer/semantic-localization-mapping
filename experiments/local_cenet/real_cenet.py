"""Offline single-scan inference; direct unprojection, no KNN or ROS."""
from pathlib import Path
import json
import time
import numpy as np

def project(points):
    if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():
        raise ValueError('Expected finite XYZI')
    depth=np.linalg.norm(points[:,:3],axis=1)
    if np.any(depth<=0): raise ValueError('Zero-range point')
    yaw=-np.arctan2(points[:,1],points[:,0])
    pitch=np.arcsin(np.clip(points[:,2]/depth,-1,1))
    px=np.clip(np.floor(.5*(yaw/np.pi+1)*2048),0,2047).astype(np.int32)
    py=np.clip(np.floor((1-(pitch+np.deg2rad(25))/np.deg2rad(28))*64),0,63).astype(np.int32)
    # Select nearest point per pixel explicitly; preserve original point indexing.
    order=np.argsort(depth,kind='stable')
    _,first=np.unique(py[order]*2048+px[order],return_index=True)
    selected=order[first]
    image=np.full((5,64,2048),-1,dtype=np.float32)
    image[:,py[selected],px[selected]]=np.column_stack((depth,points))[selected].T
    mask=np.zeros((64,2048),dtype=bool)
    mask[py[selected],px[selected]]=True
    return image,mask,px,py

def main():
    import torch
    import yaml
    root=Path(__file__).resolve().parent
    cfg=yaml.safe_load((root/'source/CENet/CENET/arch_cfg.yaml').read_text())['dataset']['sensor']
    labels=yaml.safe_load((root/'source/CENet/CENET/data_cfg.yaml').read_text())
    means=np.array(cfg['img_means'],dtype=np.float32)[:,None,None]
    std=np.array(cfg['img_stds'],dtype=np.float32)[:,None,None]
    inverse=np.array([labels['learning_map_inv'][i] for i in range(20)],dtype=np.uint32)
    out=root/'results/real06'
    (out/'predictions').mkdir(parents=True,exist_ok=True)
    model=torch.jit.load(str(root/'source/CENet/CENET/resnet34_traced_forC.pt'),map_location='cuda').eval().half()
    files=sorted((root/'data/dataset/sequences/06/velodyne').glob('*.bin'))
    rows=[]
    hist=np.zeros(20,dtype=np.int64)
    with torch.inference_mode():
        for index,f in enumerate(files):
            start=time.perf_counter()
            points=np.fromfile(f,dtype='<f4').reshape(-1,4)
            raw,mask,px,py=project(points)
            normalized=((raw-means)/std)*mask[None]
            x=torch.from_numpy(normalized).unsqueeze(0).to(device='cuda',dtype=torch.float16)
            if index==0:
                for _ in range(10): model(x)
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            infer_start=time.perf_counter()
            result=model(x)
            torch.cuda.synchronize()
            inference_ms=(time.perf_counter()-infer_start)*1000
            prob=(result[0] if isinstance(result,(tuple,list)) else result)[0].float()
            assert prob.shape==(20,64,2048) and torch.isfinite(prob).all()
            sum_error=float((prob.sum(0)-1).abs().max())
            assert sum_error<.01, 'Expected existing softmax probabilities'
            confidence,pred=prob.max(0)
            pred=pred.cpu().numpy().astype(np.uint8)
            confidence=confidence.cpu().numpy()
            per_point=pred[py,px]
            raw_labels=inverse[per_point]
            assert len(raw_labels)==len(points)
            raw_labels.astype('<u4').tofile(out/'predictions'/f'{f.stem}.label')
            hist+=np.bincount(per_point,minlength=20)
            if index in (0,99,199):
                np.savez_compressed(out/f'{f.stem}_preview.npz',points=points,labels=raw_labels,range=raw[0],mask=mask,pred=pred)
            rows.append({'frame':f.stem,'points':len(points),'occupied_pixels':int(mask.sum()),
                         'inference_ms':inference_ms,'pipeline_ms':(time.perf_counter()-start)*1000,
                         'mean_point_confidence':float(confidence[py,px].mean()),'prob_sum_max_error':sum_error})
            if (index+1)%20==0: print('frames',index+1,'/',len(files),flush=True)
    summary={'frames':len(rows),'precision':'fp16','postprocess':'direct pixel unprojection, no KNN',
             'accuracy':'not evaluated: no ground truth labels',
             'median_forward_ms':float(np.median([r['inference_ms'] for r in rows])),
             'p95_forward_ms':float(np.percentile([r['inference_ms'] for r in rows],95)),
             'median_pipeline_ms_excluding_first':float(np.median([r['pipeline_ms'] for r in rows[1:]])),
             'class_point_counts':{labels['labels'][int(inverse[i])]:int(n) for i,n in enumerate(hist) if n},
             'projection_note':'Nearest per pixel; index 0 valid (upstream Python mask uses >0). No temporal fusion.',
             'records':rows}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__': main()

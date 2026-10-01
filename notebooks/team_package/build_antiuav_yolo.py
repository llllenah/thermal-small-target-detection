#!/usr/bin/env python3
"""Build a single-class 640x512 YOLO dataset from a curated manifest and RAW ZIP.

Only selected image bytes are read. Inputs are read-only. Nonempty output directories
are refused. Exit codes: 0 complete; 1 partial build with row errors; 2 fatal input error.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image
import yaml
from tqdm import tqdm

VERSION = '1.1.0'
SPLITS = ('train','val','test')
ROLES = ('positive','negative','hard_negative')
REQUIRED = {'sequence','frame_number','image_name','exist','valid_bbox','bbox_x','bbox_y','bbox_w','bbox_h','sample_role','assigned_split'}
ERROR_FIELDS = ['row','sequence','frame_number','image_name','error','detail']

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def boolean(value) -> bool:
    text=str(value).strip().lower()
    if text in ('true','1','1.0'):return True
    if text in ('false','0','0.0'):return False
    raise ValueError(f'Invalid boolean {value!r}')

def output_stem(sequence: str, frame: int) -> str:
    safe=re.sub(r'[^A-Za-z0-9_-]','_',sequence)[:80]
    key=hashlib.sha256(sequence.encode('utf-8')).hexdigest()[:16]
    return f'{safe}__{key}__{frame:08d}'

def load_manifest(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path,dtype={'sequence':str,'image_name':str,'sample_role':str,'assigned_split':str})
    missing=REQUIRED-set(df.columns)
    if missing:raise ValueError(f'Manifest missing columns: {sorted(missing)}')
    if df.empty:raise ValueError('Manifest is empty')
    for col in ['sequence','image_name','sample_role','assigned_split']:
        if df[col].isna().any() or df[col].str.strip().eq('').any():raise ValueError(f'Blank {col}')
    if not set(df.assigned_split)<=set(SPLITS):raise ValueError('Only train/val/test splits are allowed')
    if not set(df.sample_role)<=set(ROLES):raise ValueError('Unknown sample_role')
    for col in ['frame_number','exist']:
        vals=pd.to_numeric(df[col],errors='raise')
        if vals.isna().any() or not vals.map(lambda x:math.isfinite(x) and x==int(x)).all():raise ValueError(f'Invalid integer {col}')
        df[col]=vals.astype('int64')
    if (df.frame_number<1).any() or not set(df.exist)<={0,1}:raise ValueError('Invalid frame_number or exist')
    if df.duplicated(['sequence','frame_number']).any() or df.duplicated(['sequence','image_name']).any():raise ValueError('Duplicate frame keys')
    if (df.groupby('sequence').assigned_split.nunique()>1).any():raise ValueError('Sequence appears in multiple splits')
    if 'recording_group' in df:
        if df.recording_group.isna().any() or df.recording_group.astype(str).str.strip().eq('').any():raise ValueError('Blank recording_group')
        if (df.groupby('recording_group').assigned_split.nunique()>1).any():raise ValueError('Recording group appears in multiple splits')
    for row in df.itertuples():
        if '/' in row.sequence or '\\' in row.sequence or row.sequence in ('.','..'):raise ValueError('Sequence must be a directory component')
        if PurePosixPath(row.image_name).name!=row.image_name or '\\' in row.image_name or not row.image_name.lower().endswith(('.jpg','.jpeg')):raise ValueError('image_name must be a JPEG basename')
    df['valid_bbox']=df.valid_bbox.map(boolean)
    return df.sort_values(['assigned_split','sequence','frame_number']).reset_index(drop=True)

def index_zip(z: zipfile.ZipFile, sequences: set[str]):
    index=defaultdict(list)
    # ZIP central-directory metadata only. No unselected image is opened/extracted.
    for info in tqdm(z.infolist(),desc='Index ZIP',unit='entries'):
        if info.is_dir():continue
        parts=PurePosixPath(info.filename.replace('\\','/')).parts
        if not parts or not parts[-1].lower().endswith(('.jpg','.jpeg')):continue
        hits=sequences.intersection(parts[:-1])
        for seq in hits:index[(seq,parts[-1].lower())].append((info,parts))
    return index

def resolve_member(index, row):
    candidates=index.get((row.sequence,row.image_name.lower()),[])
    if not candidates:raise FileNotFoundError('Selected JPG not found under sequence directory')
    thermal=[c for c in candidates if any(p.lower() in {'infrared','ir','thermal'} for p in c[1][:-1])]
    if thermal:candidates=thermal
    else:
        candidates=[c for c in candidates if not any(p.lower() in {'visible','rgb','color'} for p in c[1][:-1])]
    if len(candidates)>1 and hasattr(row,'original_split'):
        same=[c for c in candidates if str(row.original_split).lower() in [p.lower() for p in c[1][:-1]]]
        if same:candidates=same
    if len(candidates)!=1:raise ValueError(f'Ambiguous or non-thermal JPG match: {len(candidates)} candidates')
    return candidates[0][0]

def make_label(row, width: int, height: int) -> str:
    if row.sample_role in ('negative','hard_negative'):
        if row.exist!=0 or row.valid_bbox:raise ValueError('Negative role contradicts exist/valid_bbox')
        return ''
    if row.exist!=1 or not row.valid_bbox:raise ValueError('Invalid positive bbox or contradictory exist')
    x,y,w,h=(float(getattr(row,f'bbox_{k}')) for k in ('x','y','w','h'))
    if not all(math.isfinite(v) for v in (x,y,w,h)):raise ValueError('Nonfinite bbox')
    if w<=0 or h<=0 or x<0 or y<0 or x+w>width or y+h>height:raise ValueError(f'BBox outside {width}x{height}: {(x,y,w,h)}')
    norm=((x+w/2)/width,(y+h/2)/height,w/width,h/height)
    if not all(0<=v<=1 for v in norm):raise ValueError('Normalized bbox outside [0,1]')
    return '0 '+' '.join(f'{v:.10f}' for v in norm)+'\n'

def preflight(source, manifest, df, out) -> int:
    """Validate every selected reference using ZIP metadata; no JPEG payloads read."""
    errors=[]; seen=set(); stems=set(); image_bytes=0
    try:
        with zipfile.ZipFile(source,'r') as z:
            index=index_zip(z,set(df.sequence))
            for number,row in enumerate(tqdm(df.itertuples(index=False),total=len(df),desc='Check references',unit='frames'),start=2):
                try:
                    make_label(row,640,512)
                    info=resolve_member(index,row)
                    if info.header_offset in seen:raise ValueError('Same RAW member referenced twice')
                    stem=output_stem(row.sequence,row.frame_number)
                    if stem in stems:raise ValueError('Output name collision')
                    if info.flag_bits & 1:raise ValueError('Encrypted selected ZIP member')
                    if info.file_size==0:raise ValueError('Empty selected JPEG member')
                    if info.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED,zipfile.ZIP_BZIP2,zipfile.ZIP_LZMA}:raise ValueError('Unsupported ZIP compression')
                    seen.add(info.header_offset);stems.add(stem);image_bytes+=info.file_size
                except (ValueError,OSError) as exc:
                    errors.append({'row':number,'sequence':row.sequence,'frame_number':row.frame_number,'image_name':row.image_name,'error':type(exc).__name__,'detail':str(exc)})
    except (OSError,zipfile.BadZipFile,RuntimeError) as exc:
        errors.append({'row':'','sequence':'','frame_number':'','image_name':'','error':type(exc).__name__,'detail':'Fatal archive error: '+str(exc)})
    with (out/'preflight_errors.csv').open('w',encoding='utf-8',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=ERROR_FIELDS);writer.writeheader();writer.writerows(errors)
    report={'builder_version':VERSION,'status':'metadata_pass' if not errors else 'failed','source':str(source),'source_bytes':source.stat().st_size,'manifest_sha256':digest(manifest),'expected_frames':len(df),'matched_frames':len(seen),'errors':len(errors),'selected_image_bytes':image_bytes,'selected_image_gib':round(image_bytes/(1024**3),3),'expected_output_files':2*len(df)+3,'by_split':df.assigned_split.value_counts().to_dict(),'by_role':df.sample_role.value_counts().to_dict(),'limitations':['ZIP metadata only: JPEG dimensions, decoding, CRC and EXIF are checked during full build.','Image bytes exclude labels, filesystem allocation overhead and report files. Allow additional free space.','No annotation-completeness or cross-sequence image-duplicate audit performed.']}
    (out/'preflight_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"{report['status'].upper()}: {len(seen)}/{len(df)} references, {len(errors)} errors; images {report['selected_image_gib']} GiB. Report: {out/'preflight_report.json'}")
    return 0 if not errors else 1

def run(args) -> int:
    source=args.source.resolve();manifest=args.manifest.resolve();out=args.output.resolve()
    if not source.is_file() or not manifest.is_file():raise ValueError('Source ZIP and manifest must be existing files')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('Output must be absent or an empty directory; refusing overwrite')
    df=load_manifest(manifest)
    out.mkdir(parents=True,exist_ok=True)
    if getattr(args,'check_only',False):return preflight(source,manifest,df,out)
    report={'builder_version':VERSION,'source':str(source),'source_bytes':source.stat().st_size,'source_sha256':digest(source) if args.hash_source else None,'manifest':str(manifest),'manifest_sha256':digest(manifest),'expected_frames':len(df),'dimensions':[640,512],'class_names':{'0':'thermal_target'},'status':'building','exported_frames':0,'errors':0,'by_split':{},'by_role':{},'notes':['Strict 640x512; no resizing, clipping, or silent annotation repair.','Positive boxes invalid or out of bounds are skipped, never converted to negative.','Thermal directory preferred; ambiguous matches skipped.','Only selected JPEG payloads read; ZIP central directory indexed.']}
    errors=[];counts=Counter();roles=Counter();seen_members=set();seen_output=set()
    for split in SPLITS:
        for category in ('images','labels'):(out/category/split).mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(source,'r') as z:
            index=index_zip(z,set(df.sequence))
            for number,row in enumerate(tqdm(df.itertuples(index=False),total=len(df),desc='Build YOLO',unit='frames'),start=2):
                image_path=label_path=None
                try:
                    # Validate label semantics before opening an image, including invalid flags.
                    label=make_label(row,640,512)
                    info=resolve_member(index,row)
                    if info.header_offset in seen_members:raise ValueError('Same RAW member referenced twice')
                    raw=z.read(info)
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.format!='JPEG':raise ValueError('Payload is not JPEG')
                        if im.size!=(640,512):raise ValueError(f'Expected 640x512, got {im.size}')
                        if im.mode not in ('L','RGB'):raise ValueError(f'Unsupported image mode {im.mode}')
                        if im.getexif().get(274,1)!=1:raise ValueError('Nontrivial EXIF orientation; coordinate interpretation requires review')
                        im.load() # decode completely; corrupt/truncated JPEG is an error
                    stem=output_stem(row.sequence,row.frame_number)
                    if stem in seen_output:raise ValueError('Output name collision')
                    image_path=out/'images'/row.assigned_split/(stem+'.jpg')
                    label_path=out/'labels'/row.assigned_split/(stem+'.txt')
                    with image_path.open('xb') as dest:dest.write(raw)
                    with label_path.open('x',encoding='utf-8',newline='\n') as dest:dest.write(label)
                    seen_output.add(stem);seen_members.add(info.header_offset)
                    counts[row.assigned_split]+=1;roles[row.sample_role]+=1
                except (ValueError,OSError,zipfile.BadZipFile,RuntimeError,EOFError) as exc:
                    # Only paths created by this row, under an initially empty output, are cleaned.
                    for target in (image_path,label_path):
                        if target is not None and target.is_file():target.unlink()
                    errors.append({'row':number,'sequence':row.sequence,'frame_number':row.frame_number,'image_name':row.image_name,'error':type(exc).__name__,'detail':str(exc)})
    except (OSError,zipfile.BadZipFile,RuntimeError) as exc:
        errors.append({'row':'','sequence':'','frame_number':'','image_name':'','error':type(exc).__name__,'detail':'Fatal archive error: '+str(exc)})
    with (out/'build_errors.csv').open('w',encoding='utf-8',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=ERROR_FIELDS);writer.writeheader();writer.writerows(errors)
    yaml_data={'path':out.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':{0:'thermal_target'}}
    (out/'data.yaml').write_text(yaml.safe_dump(yaml_data,sort_keys=False,allow_unicode=True),encoding='utf-8')
    report.update(exported_frames=sum(counts.values()),errors=len(errors),by_split={s:counts[s] for s in SPLITS},by_role={r:roles[r] for r in ROLES},status='complete' if not errors and sum(counts.values())==len(df) else 'partial')
    report['file_counts']={s:{'images':len(list((out/'images'/s).glob('*.jpg'))),'labels':len(list((out/'labels'/s).glob('*.txt')))} for s in SPLITS}
    (out/'build_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"{report['status'].upper()}: {report['exported_frames']}/{len(df)} frames, {len(errors)} errors. Report: {out/'build_report.json'}")
    return 0 if report['status']=='complete' else 1

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source',required=True,type=Path,help='Read-only AntiUAV410_RAW.zip')
    p.add_argument('--manifest',required=True,type=Path,help='selected_frames.csv or .csv.gz')
    p.add_argument('--output',required=True,type=Path,help='New or empty output directory')
    p.add_argument('--hash-source',action='store_true',help='Optionally stream entire ZIP to compute SHA-256; adds a full-file read')
    p.add_argument('--check-only',action='store_true',help='Check all selected ZIP paths and label semantics without reading JPEG payloads; write preflight reports only. Use a separate empty output directory.')
    args=p.parse_args()
    if args.check_only and args.hash_source:p.error('--hash-source cannot be combined with metadata-only --check-only')
    try:return run(args)
    except (ValueError,OSError,KeyError) as exc:
        print(f'FATAL: {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())

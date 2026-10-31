"""Read-only export of historical configuration/metrics; no image or weight reads."""
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import tarfile
import time


EXPORT_SCRIPT = r'''
import collections,io,json,pathlib,re,sys,tarfile

class MetadataExporter:
    def __init__(self, project, remote):
        self.project,self.remote=pathlib.Path(project),pathlib.Path(remote)
        self.files={}
        self.total=0

    def clean(self,value):
        if isinstance(value,dict):
            return {key:self.clean(item) for key,item in value.items()
                    if key not in {'preview','previews','input_image','target_image','prediction_image'}
                    and not (isinstance(item,str) and any(s in key.lower() for s in ['token','secret','password','api_key','url']))}
        if isinstance(value,list):return [self.clean(item) for item in value]
        if isinstance(value,str):
            if value.startswith('data:image/'):return '[image omitted]'
            return re.sub(r'hf_[A-Za-z0-9]{20,}', '[token omitted]', value)
        return value

    def add(self,name,value):
        content=json.dumps(self.clean(value),ensure_ascii=False).encode('utf-8')
        self.total+=len(content)
        if self.total>25000000:raise RuntimeError('Metadata export exceeds 25 MB')
        self.files[name]=content

    def read(self,path,name):
        if not path.is_file() or path.is_symlink():return
        if path.stat().st_size>12000000:raise RuntimeError('Oversized metadata: '+str(path))
        try:
            if path.suffix=='.jsonl':
                values=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                self.add(name[:-1],values) # metrics.jsonl -> metrics.json
            else:self.add(name,json.loads(path.read_text(encoding='utf-8-sig')))
        except ValueError as error:self.add(name+'.error.json',{'error':str(error)})

    def runs(self,base,prefix):
        if not base.is_dir():return
        for run in sorted(base.iterdir()):
            if not run.is_dir() or run.is_symlink():continue
            if run.name=='adapter_loss':
                for job in sorted(run.iterdir()):
                    if job.is_dir():
                        for name in ['job.json','request.json']:
                            self.read(job/name,prefix+'/adapter_loss/'+job.name+'/'+name)
                continue
            for name in ['config.json','job.json','metrics.jsonl','weights/manifest.json','weights/encoder_lora_config.json']:
                self.read(run/name,prefix+'/'+run.name+'/'+name)
            weights=run/'weights'
            if weights.is_dir():
                self.add(prefix+'/'+run.name+'/weights_inventory.json',
                         [{'name':str(p.relative_to(weights)),'bytes':p.stat().st_size}
                          for p in weights.rglob('*') if p.is_file() and not p.is_symlink()])

    def dataset_manifest(self):
        datasets=[]
        base=self.project/'datasets'
        if not base.is_dir():return
        extensions={'.png','.jpg','.jpeg','.webp','.bmp'}
        for dataset in sorted(base.iterdir()):
            if not dataset.is_dir():continue
            splits={}
            for split in ['train','val','test']:
                directory=dataset/split
                if not directory.is_dir():continue
                starts,ends={},{}
                for p in directory.rglob('*'):
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in extensions:
                        key=str(p.relative_to(directory).with_suffix(''))
                        if key.endswith('_start'):starts[key[:-6]]=p
                        if key.endswith('_end'):ends[key[:-4]]=p
                pairs=[]
                for key in sorted(starts.keys() & ends.keys()):
                    p=starts[key]
                    captions=[p.with_suffix('.txt'),p.with_name(p.stem[:-6]+'.txt')]
                    text=next((c.read_text(encoding='utf-8',errors='replace').strip() for c in captions if c.is_file() and not c.is_symlink()),'')
                    pairs.append({'key':key,'caption_chars':len(text)})
                splits[split]={'pair_count':len(pairs),'missing_end':sorted(starts.keys()-ends.keys()),
                               'missing_start':sorted(ends.keys()-starts.keys()),'pairs':pairs}
            datasets.append({'dataset':dataset.name,'splits':splits,
                             'note':'Only filenames and caption lengths inspected; no image bytes opened.'})
        self.add('datasets_manifest.json',datasets)

    def export(self):
        self.runs(self.project/'runs','legacy')
        jobs=self.remote/'jobs'
        if jobs.is_dir():
            for job in sorted(jobs.iterdir()):
                if job.is_dir():self.runs(job/'workspace'/'runs','remote/'+job.name)
        self.dataset_manifest()
        models=self.project/'models'
        if models.is_dir():
            self.add('models_inventory.json',[{'name':p.name,'bytes':p.stat().st_size} for p in models.iterdir() if p.is_file()])
            for p in models.glob('*.json'):self.read(p,'models/'+p.name)
        self.add('export_manifest.json',{'files':list(self.files),'total_bytes':self.total,'images_read':False,'weights_read':False})
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
            for name,content in self.files.items():
                entry=tarfile.TarInfo(name)
                entry.size=len(content)
                archive.addfile(entry,io.BytesIO(content))

MetadataExporter(sys.argv[1],sys.argv[2]).export()
'''


class HistoryCollector:
    def __init__(self, settings, destination):
        self.settings, self.destination = settings, Path(destination)

    def collect(self, ssh):
        content = ssh.run(shlex.join(['python3', '-c', EXPORT_SCRIPT,
                                      self.settings.remote_project, self.settings.remote_base]), timeout=90)
        self.destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts or ':' in member.name or '\\' in member.name:
                    raise ValueError('Unsafe historical metadata path')
                if member.size > 25_000_000:
                    raise ValueError('Oversized historical metadata')
                target = self.destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        return self.destination

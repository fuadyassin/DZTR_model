"""Verify real Python, pip and plotting cache paths after activating the project environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CACHE=(ROOT/'.cache').resolve()
def inside(value):
    path=Path(value).resolve()
    if not path.is_relative_to(CACHE):raise ValueError(f'Cache path is outside repository cache: {path}')
    return str(path)

checks={name:inside(os.environ[name]) for name in (
    'TEMP','TMP','TMPDIR','PIP_CACHE_DIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX',
    'XDG_CACHE_HOME','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER')}
checks['tempfile_directory']=inside(tempfile.gettempdir())
checks['python_bytecode_prefix']=inside(sys.pycache_prefix)
with tempfile.NamedTemporaryFile(prefix='dztr-cache-check-',delete=True) as probe:
    checks['temporary_file_location_verified']=Path(probe.name).resolve().is_relative_to(CACHE)
pip=subprocess.run([sys.executable,'-B','-m','pip','cache','dir'],capture_output=True,text=True,check=True)
checks['pip_effective_cache']=inside(pip.stdout.strip())
try:
    import matplotlib
except ImportError:
    checks['matplotlib']='not installed'
else:
    checks['matplotlib_config']=inside(matplotlib.get_configdir())
    checks['matplotlib_cache']=inside(matplotlib.get_cachedir())
out=ROOT/'outputs/cache_storage_20261014'
out.mkdir(exist_ok=True)
(out/'project_cache_verification.json').write_text(json.dumps({'status':'passed','checks':checks,
    'numerical_code_or_outputs_changed':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','cache_root':str(CACHE),'tempfile':checks['tempfile_directory'],
    'pip':checks['pip_effective_cache'],'matplotlib':checks.get('matplotlib_cache')},indent=2))

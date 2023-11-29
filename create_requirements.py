import subprocess
import sys

# Execute pip freeze and write the output to requirements.txt
with open('requirements.txt', 'w') as f:
    subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'], stdout=f)
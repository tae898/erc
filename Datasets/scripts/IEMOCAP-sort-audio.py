import os
from glob import glob

os.chdir('IEMOCAP/raw-audios')
    
for filename in glob('./*.wav'):
    DIR = '_'.join(filename.split('_')[:-1])
    os.makedirs(DIR, exist_ok=True)
    os.rename(filename, os.path.join(DIR, filename))

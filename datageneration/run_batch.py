import os
from os.path import join
import subprocess

with open(join(os.path.expanduser('~'),'Documents','MVA','surreal','datageneration','misc','job_list.txt'),'r') as f:
    list_job = [line.split('\n')[0] for line in f]

sdir = join(os.path.expanduser('~'),'Documents','MVA','datageneration')

for i,job_params in enumerate(list_job):
    if (i<20):
	print('current job: %s' %job_params)
        cmd = subprocess.call(sdir +'/' +'run.sh' +' '+ '\''+job_params+'\'', shell=True)        
        with open(join(os.path.expanduser('~'),'Documents','MVA','surreal','datageneration','misc','job_done.txt'),'a') as f:
            f.write(job_params)
            

            
        
    
    




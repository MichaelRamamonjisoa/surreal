import os
from os.path import join
import subprocess

with open(join(os.path.getcwd(), 'misc', 'job_list.txt'),'r') as f:
    list_job = [line.split('\n')[0] for line in f]

sdir = os.path.getcwd()

with open(join(os.path.getcwd(),'misc','job_done.txt'),'r') as f:
    job_done = [line.split('\n')[0] for line in f]
    
nb_done = len(job_done)

for i,job_params in enumerate(list_job):    
    if i>nb_done-1 and i<nb_done + 109:
	print('current job: %s' %job_params)
        cmd = subprocess.call(sdir +'/' +'run.sh' +' '+ '\''+job_params+'\'', shell=True)
        with open(join(os.path.getcwd(),'misc','job_done.txt'),'a') as f:
            f.write(job_params+'\n')
            

            
        
    
    




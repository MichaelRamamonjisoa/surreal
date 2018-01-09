import os
from os.path import join
import subprocess
import sys

output_part1 = 'job_done_part1.txt'
output_part2 = 'job_done_part2.txt'

def help():
    print("./run_batch.py (part1 | part2) runs")
    exit(-1)

if len(sys.argv) < 3:
    help()

if sys.argv[1] == 'part1':
    output = output_part1
    reverse = False
elif sys.argv[1] == 'part2':
    output = output_part2
    reverse = True
else:
    help()

try:
    runs = int(sys.argv[2])
except ValueError:
    help()

with open(join(os.getcwd(), 'misc', 'job_list.txt'),'r') as f:
    list_job = [line.split('\n')[0] for line in f]

sdir = os.getcwd()

with open(join(os.getcwd(),'misc', output_part1),'r') as f:
    job_done_part1 = [line.split('\n')[0] for line in f]

with open(join(os.getcwd(),'misc', output_part2),'r') as f:
    job_done_part2 = [line.split('\n')[0] for line in f]

job_done = set(job_done_part1).union(set(job_done_part2))

if reverse:
    list_job = reversed(list_job)

i = 0
for job_params in list_job:
    if job_params not in job_done:
        print('current job:', job_params)
        cmd = subprocess.call(sdir +'/run.sh' +' "' + job_params +'"', shell=True)
        with open(join(os.getcwd(),'misc',output),'a') as f:
            f.write(job_params+'\n')
        i += 1
        if (i >= runs): break

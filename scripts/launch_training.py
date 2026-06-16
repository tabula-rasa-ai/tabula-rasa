"""Launch training in background with proper error logging."""
import sys, os, subprocess, time

log_file = r'C:\Users\Admin\tabula-rasa\launch.log'
cmd = r'cd C:\Users\Admin\tabula-rasa && python3 -u train_specialist.py add --steps 50000 --batch 256 --lr 5e-4'

with open(log_file, 'w') as f:
    f.write(f'Launching at {time.ctime()}: {cmd}\n')

proc = subprocess.Popen(
    cmd,
    shell=True,
    stdout=open(log_file, 'a'),
    stderr=subprocess.STDOUT,
)

with open(log_file, 'a') as f:
    f.write(f'PID: {proc.pid}\n')

print(f'Training launched. PID: {proc.pid}')
print(f'Log: {log_file}')
print(f'Monitor: tail -f {log_file}')

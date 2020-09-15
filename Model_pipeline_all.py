from subprocess import run, PIPE, STDOUT
import time
#import atexit

# Initialising time:
start_time = time.time()

countries = 'Denmark|Germany|Spain|Finland|Italy|Sweden|France|Norway|United Kingdom' \
            '|United States|Canada|Mexico' \
            '|Australia|Indonesia|Malaysia|Philippines|Thailand|Vietnam|China|India|Japan|Singapore|Taiwan' \
            '|Saudi Arabia|United Arab Emirates'

#procs = []

for i in countries.split('|'):
    start_timeone = time.time()
    proc = run(["python", ".\Model_pipeline_one.py", i], stdout=PIPE, stderr=STDOUT, universal_newlines=True)
    runtimeone = time.gmtime(time.time() - start_timeone)
    resone = time.strftime('%M:%S', runtimeone)
    print('************* Model for {} created in {} mins/secs'.format(i, resone))
#    print(proc)

#    procs.append(proc)
#    proc.kill()
#    proc.terminate()
#    proc = subprocess.Popen(["python", ".\Model_pipeline_one.py", i], stdin=PIPE, stdout=PIPE)
#    try:
#        outs, errs = proc.communicate(timeout=50)
#    except TimeoutExpired:
#        proc.kill()
#        outs, errs = proc.communicate()


#@atexit.register
#def kill_subprocesses():
#    for proc in procs:
#        proc.kill()


runtime = time.gmtime(time.time() - start_time)
res = time.strftime('%M:%S', runtime)
print('************* All models created in {} mins/secs'.format(res))
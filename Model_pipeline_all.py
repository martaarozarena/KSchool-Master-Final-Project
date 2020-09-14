import subprocess

countries = 'Denmark|Germany|Spain|Finland|Italy|Sweden|France|Norway|United Kingdom' \
            '|United States|Canada|Mexico' \
            '|Australia|Indonesia|Malaysia|Philippines|Thailand|Vietnam|China|India|Japan|Singapore|Taiwan' \
            '|Saudi Arabia|United Arab Emirates'

for i in countries.split('|'):
    proc = subprocess.Popen(["python", ".\Model_pipeline_one.py", i])
#    proc.kill()

#    try:
#        outs, errs = proc.communicate(timeout=50)
#    except TimeoutExpired:
#        proc.kill()
#        outs, errs = proc.communicate()

proc.kill()
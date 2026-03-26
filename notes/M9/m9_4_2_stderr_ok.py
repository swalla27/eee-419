# Example to show stdout message returned by subprocess

import subprocess      # package to run commands

proc = subprocess.Popen(["ls","-F"],              # command and args
                         stdout=subprocess.PIPE,  # redirect stdout
                         stderr=subprocess.PIPE)  # redirect stderr
output, err = proc.communicate()
print("output is\n",output)
print("err is\n",err)



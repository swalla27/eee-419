# Example to show error message returned by subprocess

import subprocess      # package to run commands

proc = subprocess.Popen(["cp","nofile newfile"],   # command and args
                         stdout=subprocess.PIPE,   # redirect stdout
                         stderr=subprocess.PIPE)   # redirect stdin
output, err = proc.communicate()
print("output is\n",output)
print("err is\n",err)

# #################
# #### Code 1: Copy file through Terminal
# import shutil           # package needed to copy a file

# shutil.copy("m9_4_1_my_copy.py","copied.py")  # call the utility to do the copy

##################
#### Code 2: Example to show stdout message returned by subprocess

# import subprocess      # package to run commands

# # Windows:
# proc = subprocess.Popen(["cmd","/c","dir"],       # command and args
#                          stdout=subprocess.PIPE,  # redirect stdout
#                          stderr=subprocess.PIPE)  # redirect stderr

# # # Unix (Linux/Mac):
# # # proc = subprocess.Popen(["ls","-l"],              # command and args
# # #                          stdout=subprocess.PIPE,  # redirect stdout
# # #                          stderr=subprocess.PIPE)  # redirect stderr
# output, err = proc.communicate()
# # print("output is\n",output)         # output is a bytes object b''
# # print("err is\n",err)               # err is a bytes object b''
# print("output is\n",output.decode('utf-8'))
# print("err is\n",err.decode('utf-8'))
# print('Hello World')

###################
# #### # Create Directory

# import subprocess      # package to run commands

# # Windows:
# proc_mkdir = subprocess.Popen(["cmd","/c","mkdir","NewDir"],   # command and args
#                          stdout=subprocess.PIPE,   # redirect stdout
#                          stderr=subprocess.PIPE)   # redirect stdin

# # # Mac:
# # proc = subprocess.Popen(["mkdir","NewDir"],   # command and args
# #                          stdout=subprocess.PIPE,   # redirect stdout
# #                          stderr=subprocess.PIPE)   # redirect stdin

# input()
# proc_rmdir = subprocess.Popen(["cmd", "/c", "rmdir", "NewDir"],
#                          stdout=subprocess.PIPE,   # redirect stdout
#                          stderr=subprocess.PIPE)   # redirect stdin)
# output, err = proc_rmdir.communicate()
# print("output is\n",output)
# print("err is\n",err)



########## Automation:
# # Write a line to a file
# # open('My_File_2.txt', 'a').write('I wrote this line2\n')
# with open('My_File_1.txt', 'a') as write_file1:
#     write_file1.write('I wrote this line\n')

##################################

# # # Append a line to the end of a file
# with open('My_File_write.txt', 'a') as read_file:
#     read_file.write('I appended this line\n')

# ################################
# # Write the items of a list to a txt file
# x_list = ["2A\n"] + ["3B\n"]                    # the same as x_list = ["2A", "3B"]
# print(x_list)
# with open('My_File_1.txt', 'a') as write_file:
#     write_file.writelines(x_list)         # each item in x_list must be a string 

# #################################
# # # Read a file, copy its lines to another file:
# ### Opening both files:
# with open('My_File_2.txt', 'a') as write_to_file, open('My_File_1.txt', 'r') as read_from_file:
#     ### Reading the lines from one of them:
#     lines_read = read_from_file.readlines()
#     print(lines_read)
#     ### Writing them to the other
#     write_to_file.writelines(lines_read)


# ####################################
# # # Read a file, copy its contents to another file,
# # # append 10 more lines, each on the form of "Additional Line i" where i is the index of the added line
# ### Opening both files:
# with open('My_File_3.txt', 'w') as write_to_file, open('My_File_2.txt', 'r') as read_from_file:
#     ### Reading the lines from one of them:
#     lines_read = read_from_file.readlines()
#     print(lines_read)
#     ### Writing them to the other
#     write_to_file.writelines(lines_read + [f'Additional Line {i}\n' for i in range(1,11)])
# ### Another implementation by writing the fixed part then appending the dynamic part:
# # with open('My_File_write.txt', 'w') as write_to_file, open('My_File_read.txt', 'r') as read_from_file:
# #     ### Reading the lines from one of them:
# #     lines_read = read_from_file.readlines()
# #     print(lines_read)
# #     ### Writing them to the other
# #     write_to_file.writelines(lines_read)
# # with open('My_File_write.txt', 'a') as append_to_file:
# #     for i in range(1,11):
# #         append_to_file.write(f"Additional Line {i}\n")


# ###############
# #### Run MATLAB Code:

# # with open('testing.m', 'w') as read_file:
# #     read_file.write(r"fprintf('Hello World\n')")

# import subprocess      # package to run commands
# # matlab_dir = r'C:\Users\Ewaisha\ASU Dropbox\Ahmed Ewaisha\ASU\Teaching Classes\EEE 419\M9'
# # matlab_file = 'testing.m'

# proc = subprocess.Popen(["cmd","/c" , r"""matlab -batch "run('testing.m')" """],
#                          stdout=subprocess.PIPE,   # redirect stdout
#                          stderr=subprocess.PIPE)   # redirect stdin)
# output, err = proc.communicate()
# print("output is\n",output)
# print("err is\n",err)


##################################
####### Passing a value to MATLAB:
## How to create a function in MATLAB:

# ##### Python Function:
# # def fun_mult10(x):
# #     y = x*10
# #     return y

# # ##### Equivalent MATLAB Function (just for reference):
# # # function y = fun_mult10(x)
# # # y = x*10

# import subprocess      # package to run commands
# # # matlab_dir = r'C:\Users\Ewaisha\ASU Dropbox\Ahmed Ewaisha\ASU\Teaching Classes\EEE 419\M9'
# # # matlab_file = 'testing.m'

# proc = subprocess.Popen(["cmd","/c" , r"""matlab -batch "run('fun_mult10({i}).m')" """],   # command and args
#                  stdout=subprocess.PIPE,   # redirect stdout
#                  stderr=subprocess.PIPE)   # redirect stdin

# output, err = proc.communicate()
# print("output is\n",output.decode('UTF-8'))
# print("err is\n",err)


#####################
# ####### Running MATLAB with different iterations
# import subprocess      # package to run commands
# # # matlab_dir = r'C:\Users\Ewaisha\ASU Dropbox\Ahmed Ewaisha\ASU\Teaching Classes\EEE 419\M9'
# # # matlab_file = 'testing.m'

# x = 10
# strr = """matlab -batch "run('fun_div100("""+str(x)+""").m')" """
# print(strr)
# proc = subprocess.Popen(["cmd","/c" , strr],   # command and args
#                  stdout=subprocess.PIPE,   # redirect stdout
#                  stderr=subprocess.PIPE)   # redirect stdin

# output, err = proc.communicate()
# print("output is\n",output)
# print("err is\n",err)


###################
# #### Code : Requesting a password without displaying it to the screen while user types

# import getpass

# # Prompt the user for input, but hide what they type
# user_input = getpass.getpass("Enter your password: ")
# print("Your password is", user_input)




# # Xinv2 b c inv M=fan**1
# # Xinv3 c d inv M=fan**2
# # Xinv4 d e inv M=fan**3
# # Xinv5 e f inv M=fan**4
# # Xinv6 f g inv M=fan**5
# # Xinv7 g h inv M=fan**6
# # Xinv8 h i inv M=fan**7
# # Xinv9 i j inv M=fan**8
# # Xinv10 j k inv M=fan**9
# # Xinv11 k z inv M=fan**10

####################
# # Specify the file name and the line to append
# file_name = "InvChain.sp"
# new_line = "This is the new line I want to append.\n"

# # Open the file in append mode and write the line
# with open(file_name, 'a') as file:
#     file.write(new_line)




# # Original file to read from
# original_file = "InvChain.txt"

# # New file to save the appended content
# new_file = "InvChain.sp"

# # Line to append
# new_line = "This is the new line to add.\n"

# # Read original content, append new line, and save to a new file
# with open(original_file, "r") as file:
#     content = file.read()

# with open(new_file, "w") as file:
#     file.write(content)    # Write original content
#     file.write(new_line)   # Append the new line


# Unix:
# scp "M9.py" "aewaisha@eecad102.eas.asu.edu/~"

# Windows:
# scp M9.py "aewaisha@eecad102.eas.asu.edu:"
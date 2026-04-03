# Example SQLite program
# author charles severance updated by sdm
# reference: https://www.sqlite.org/lang.html

import sqlite3                                  # get the package

conn = sqlite3.connect('maildb.sqlite')         # open the database
cur = conn.cursor()                             # and connect to it

cur.execute('DROP TABLE IF EXISTS Counts')      # delete table if already there!

# create the table we want in our database
# Call the table Counts and it has two fields: email and count
cur.execute('''CREATE TABLE Counts (email TEXT, count INTEGER)''')

# get the data file - note the default name to use
fname = input('Enter file name: ')
if (len(fname) < 1): fname = 'mailbox.txt'
fh = open(fname)

for line in fh:                                   # for each line in the file...
    if not line.startswith('From: '): continue    # only look at From lines
    pieces = line.split()                         # chunk the line
    email = pieces[1]                             # the address is the 2nd item

    # Get value count field in the table Counts that matches the email address
    cur.execute('SELECT count FROM Counts WHERE email = ? ', (email,))

    row = cur.fetchone()                          # this tells us the row
    if row is None:                               # new email address?
        cur.execute('''INSERT INTO Counts (email, count)
                    VALUES (?, 1)''', (email,))   # if so, make a new entry

    else:                                         # else, increment its count
        cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',
                    (email,))
    conn.commit()                                 # SAVE the change

# DESC means descending order
# print out the top 10 email address found in the database
print("\nTop 10 email addresses:")
sqlstr = 'SELECT email, count FROM Counts ORDER BY count DESC LIMIT 10'
for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])

print("\nBottom 10 email addresses:")
# print out the bottom 10 email address found in the database
sqlstr = 'SELECT email, count FROM Counts ORDER BY count ASC LIMIT 10'
for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])

cur.close()     # and close the database

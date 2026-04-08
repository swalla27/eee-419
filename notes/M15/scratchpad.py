

# Option A.
with open('nuclear_codes.txt', 'w') as f:
    lines = f.readlines()
    for line in lines:
        print(line)

# Option B.
f = open('nuclear_codes.txt', 'w')
lines = f.readlines()
for line in lines:
    print(line)


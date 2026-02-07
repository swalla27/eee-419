binomial_expansion = [1, [1, 1], [1, 2, 1]]
N = 20

for i in range(3, N):
    previous_row = binomial_expansion[i-1]
    new_row = [1, 1]
    
    insert_at = 1
    print(binomial_expansion[i-1])
    for j, entry in enumerate(previous_row[:-1]):
        new_row.insert(insert_at, previous_row[j] + previous_row[j+1])

        insert_at += 1

    binomial_expansion.append(new_row)

for row in binomial_expansion:
    print(row)
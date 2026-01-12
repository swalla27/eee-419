# Program to illustrate math operations

import numpy as np      # so we can demo square root

sum = 2 + 3
dif = 8 - 1
mul = 4 * 5
div = 32 / 8
idiv = 7 // 3           # integer division
pow = 2 ** 5            # exponent
mod = 11 % 3            # modulus
sq = np.sqrt(25)        # use this square root!

print("Sum is:",sum)
print("Dif is:",dif)
print("Mul is:",mul)
print("Div is:",div)
print("Idiv is:",idiv)
print("Pow is:",pow)
print("Mod is:",mod)
print("Square root is:",sq)

input()

# convenience operators
# Note that ++ and -- are NOT supported
sum += 4
print("Sum is now:",sum)


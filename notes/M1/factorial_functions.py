# Problem 2:
# Implement the factorial function using recursion
# Then again using for loop

def factorial_recursive(x: int):
    if x == 0:
        return 1
    else:
        return x * factorial_recursive(x-1)
    
print(factorial_recursive(10))
print(factorial_recursive(3))
print(factorial_recursive(5))

def factorial_loop(x: int):
    result = 1
    if x == 0:
        return 1
    
    while x > 1:
        result = result*x
        x -= 1

    return result

print(factorial_loop(6))
import datetime

class Employee:

    # This is a class variable, do not change this for a single instance. Always do it for the entire class.
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@gmail.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    def __repr__(self):
        # A dunder method used to make a string representation of your class, programmers are the intended audience.
        return f'Employee({self.first}, {self.last}, {self.pay})'
    
    def __str__(self):
        # A dunder method used to make a string representation of your class, end user is the intended audience.
        return f'{self.fullname()} - {self.email}'
    
    def __add__(self, other):
        return self.pay + other.pay
    
    def __len__(self):
        return len(self.fullname())
    
    @classmethod
    # Do not run this from an instance, always do it from the class.
    # It accesses the class (cls) instead of the instance (self).
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    # An alternative contructor, meaning you can create an instance this way too.
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)
    
    @staticmethod
    # A static method does not pass the instance (self) or the class (cls) as input.
    # If you don't access self or cls in the function, then it should probably be a static method.
    def is_workday(day):
        if (day.weekday() == 5) or (day.weekday() == 6):
            return False
        else:
            return True

class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang

class Manager(Employee):

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)

        if employees is None:
            self.employees = []
        else:
            self.employees = employees

        self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
    
    def rem_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())

emp_1 = Employee('Corey', 'Shafer', 50_000)
emp_2 = Employee('Test', 'Employee', 60_000)

emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'

new_emp_1 = Employee.from_string(emp_str_1)
print(new_emp_1.fullname())

print(Employee.is_workday(datetime.date(2016, 7, 11)))


# print(Employee.raise_amt)

# Employee.set_raise_amt(1.20)

# print(emp_1.raise_amt)
# print(emp_2.raise_amt)
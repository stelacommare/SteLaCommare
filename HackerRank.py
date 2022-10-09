#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############ INTRODUCTION


# In[ ]:


if __name__ == '__main__':
    print("Hello, World!")


# In[ ]:


import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 1:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
else:
    print("Not Weird")


# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)


# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)


# In[ ]:


if __name__ == '__main__':
    n = int(input())
i=0
while i<n:
    print(i*i)
    i+=1


# In[ ]:


def is_leap(year):
    leap = False
    if year%4 == 0:
        leap = True
        if year%100 == 0:
            leap = False
            if year%400 == 0:
                leap = True
    
    return leap

year = int(input())


# In[ ]:


if __name__ == '__main__':
    n = int(input())
i=1
a=""
while i<=n:
    a += str(i)
    i += 1
print(a)


# In[ ]:


################# DATA TYPES


# In[ ]:


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k!=n ])


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    max = max(arr)
    score = None

    for num in arr:
        if num == max:
            pass
        elif score == None:
            score = num
        elif num > score:
            score = num

    print(score)


# In[ ]:


Result =[]
scorelist = []
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        Result+=[[name,score]]
        scorelist+=[score]
    b=sorted(list(set(scorelist)))[1] 
    for a,c in sorted(Result):
        if c==b:
            print(a)


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    import builtins
    t=tuple(integer_list)
    print(hash(t))


# In[ ]:


################### STRINGS


# In[ ]:


def swap_case(s):
    return s.swapcase()
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# In[ ]:


def split_and_join(line):
    line = "-".join(line.split(" "))
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# In[ ]:


def print_full_name(first, last):
    print("Hello",first,last+"!","You just delved into python.")
    # Write your code here

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# In[ ]:


def count_substring(string, sub_string):
    count=0
    for i in range(0, len(string)-len(sub_string)+1):
        if string[i] == sub_string[0]:
            flag=1
            for j in range (0, len(sub_string)):
                if string[i+j] != sub_string[j]:
                    flag=0
                    break
            if flag==1:
                count += 1
    return count


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


# In[ ]:


if __name__ == '__main__':
    s = input()
    print(any(map(str.isalnum, s)))
    print(any(map(str.isalpha, s)))
    print(any(map(str.isdigit, s)))
    print(any(map(str.islower, s)))
    print(any(map(str.isupper, s)))


# In[2]:


if __name__ == '__main__':
    N = int(input())
    m=list()
    for i in range(N):
       method,*l=input().split()
       k=list(map(int,l))
       if len(k)==2:
          q=[k[0]]
          w=[k[1]]
       elif len(k)==1:
          q=[k[0]]
       if method =='insert':
          m.insert(q[0],w[0])
       elif method == 'append':
          m.append(q[0])
       elif  method == 'remove':
          m.remove(q[0])
       elif method =='print':
          print(m)
       elif method == 'reverse':
          m.reverse()
       elif method =='pop':
          m.pop()
       elif method == 'sort':
          m.sort()


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    sonuc = 0
    average = float()
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    for i in student_marks[query_name]:
        sonuc += i
    average = sonuc / len(student_marks[query_name])
    print("%.2f" %average)


# In[ ]:


def mutate_string(string, position, character):
    x=list(string)
    x[position] = character
    string = "".join(x)
    return string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# In[ ]:


import textwrap

def wrap(string, max_width):
    for i in range(0,len(string)+1,max_width):
        result = string[i:i+max_width]
        if len(result) == max_width:
            print(result)
        else:
            return(result)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# In[ ]:


def print_formatted(number):

    width = len("{0:b}".format(number))

    for i in range(1, number + 1):
        print("{0:{w}d} {0:{w}o} {0:{w}X} {0:{w}b}".format(i, w = width))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# In[ ]:


################ SETS


# In[ ]:


def average(array):
    return sum(set(array))/len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# In[ ]:


M = int(input())
Mset = set(map(int, input().split()))
N = int(input())
Nset = set(map(int, input().split()))

Mdiff = Mset.difference(Nset)
Ndiff = Nset.difference(Mset)

output = Mdiff.union(Ndiff)

for i in sorted(list(output)):
    print(i)


# In[ ]:


N = int(input())
countr = set()
for i in range(N):
    countr.add(input())
print(len(countr))


# In[ ]:


e = int(input())
eng = set(input().split())
f = int(input())
french = set(input().split())
print(len(eng.union(french)))


# In[ ]:


n = int(input())
eng = set(input().split())
b = int(input())
french = set(input().split())
print(len(eng.intersection(french)))


# In[ ]:


n = int(input())
eng = set(input().split())
b = int(input())
french = set(input().split())
print(len(eng.difference(french)))


# In[ ]:


n = int(input())
eng = set(input().split())
b = int(input())
french = set(input().split())
print(len(eng.symmetric_difference(french)))


# In[ ]:


N = int(input())
A = set(map(int, input().split()))
for i in range(int(input())):
    command = input().split()
    b = set(map(int, input().split()))
    if command[0] == 'update':
        A.update(b)
    if command[0] == 'intersection_update':
        A.intersection_update(b)
    if command[0] == 'difference_update':
        A.difference_update(b)
    if command[0] == 'symmetric_difference_update':
        A.symmetric_difference_update(b)      
print(sum(A))


# In[ ]:


for i in range(int(input())):
    a = int(input())
    aset = set(map(int, input().split()))
    b = int(input())
    bset = set(map(int, input().split()))
    if len(aset-bset) == 0:
        print("True")
    else:
        print("False")


# In[ ]:


############### COLLECTIONS


# In[ ]:


N = int(input())
col = input().split()
info = [input().split() for _ in range(N)]
mark = col.index("MARKS")
sum = 0
for i in info:
    sum += int(i[mark])
print(round(sum/N, 2))


# In[ ]:


################ DATE AND TIME


# In[ ]:


import datetime
import calendar
m, d, y = map(int, input().split())
input_date = datetime.date(y, m, d)
print(calendar.day_name[input_date.weekday()].upper())


# In[ ]:


import math
import os
import random
import re
import sys

from datetime import datetime

def time_delta(t1, t2):
    
    first = datetime.strptime(t1,'%a %d %b %Y %H:%M:%S %z')
    second = datetime.strptime(t2,'%a %d %b %Y %H:%M:%S %z')
    return str(abs(int((first-second).total_seconds())))
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


# In[ ]:


################ BUILT-INS


# In[ ]:


x, k = list(map(int,input().split()))
P = eval(input())
print(P==k)


# In[ ]:


N, X = input().split()
y = list()
for _ in range(int(X)):
    m = map(float, input().split())
    y.append(m)
for i in zip(*y): 
    print(sum(i)/len(i)) 


# In[ ]:


################# Regex and Parsing challenges


# In[ ]:


regex_pattern = r'[.,]+'    # Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))


# In[ ]:


import re
expression=r"([a-zA-Z0-9])\1+"
x = re.search(expression,input())
if x:
    print(x.group(1))
else:
    print(-1)


# In[ ]:


################## NUMPY


# In[ ]:


import numpy

def arrays(arr):
    return numpy.array(arr[::-1],float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# In[ ]:


import numpy
n,m = map(int,input().split())
ar = []
for i in range(n):
    row = list(map(int,input().split()))
    ar.append(row)
np_ar = numpy.array(ar)
print(numpy.transpose(np_ar))
print(np_ar.flatten())


# In[ ]:


import numpy as np
N, M, P = list(map(int, input().split()))
arr1 = np.array([list(map(int, input().split())) for _ in range(N)])
arr2 = np.array([list(map(int, input().split())) for _ in range(M)])
print(np.concatenate((arr1, arr2), axis=0))


# In[ ]:


import numpy as np
nums = tuple(map(int, input().split()))
print (np.zeros(nums, dtype = np.int))
print (np.ones(nums, dtype = np.int))


# In[ ]:


import numpy as np
N, M = map(int, input().split())
A = np.array([input().split() for _ in range(N)], int)
print(np.prod(np.sum(A, axis=0), axis=0))


# In[ ]:


################### PROBLEM 2 ##########################


# In[ ]:


import math
import os
import random
import re
import sys
def birthdayCakeCandles(ar):
    # Write your code here
    count = 0 
    maxHeight = max(ar)
    for i in ar:
        if i == maxHeight:
            count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

    fptr.close()


# In[ ]:


import math
import os
import random
import re
import sys
def viralAdvertising(n):
    shared =5
    cumulative=0
    for i in range(1,n+1):
        liked = shared//2
        cumulative+=liked
        shared = liked*3
    return cumulative
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()


# In[ ]:


def superDigit(n, k):
    return 1 + (k * sum(int(x) for x in n) - 1) % 9

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()


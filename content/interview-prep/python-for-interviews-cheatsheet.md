---
title: Python for Interviews Cheatsheet
draft: false
tags:
  - Interview
  - Python
  - DataStructures
  - Algorithms
---
## Introduction
I love Python. It's simple and gets the job done. Writing Python code is generally faster and easier than the other languages because it handles most of the work in the background for you instead of making you implement it. Because of this trait, I think it suits well for coding interview questions. As a third-year computer engineering student, I must always be ready to be tested. This requires me to have a language that I can write well all the time for interview questions. I generally don't write Python or don't solve interview questions daily. Because of that, I spend more time thinking about the syntax and programming language-related things than the actual implementation of the question. However, this must be changed. I'm writing this as a cheat sheet for myself to have a look at how things are done in Python, especially for interviews. My main source for writing this was [[https://www.youtube.com/watch?v=0K_eZGS5NsU&t=685s|Python for Coding Interviews - Everything you Need to Know by NeetCode]] video. If you want to see something more detailed, I suggest you go and check it out yourself. Too much talk, let's dive in
## Cheatsheet
### Variables  
Multiple assignments can be done in this way
```python
a, b, c = 31, "abc", False
```

Unfortunately, we can't use i++

```python
n = n + 1
n += 1
# n++ # syntax error
```
We don't have NULL, but have None as equivalent
```python
n = None
```
### If Statements  
Everything is straightforward in if statements except there is no else if but there is a separate keyword as `elif`. Also, attention to indentation.
```python
n = 1
if n < 2:
	n -= 1
elif n > 69:
	n *= 2
else:
	n += 3
```
Parentheses are needed for multi-line conditions. For conditions, we write `and` instead of `&&` and `or` instead of `||`.
```python
n, m = 1, 2
if ((n > 2 and
	n != m) or n == m):
	n += 1
```
Inline `if` statements are useful for some cases
```python
result = "Positive" if number > 0 else "Non-positive"
```
### Loops  
```python
while True:
	print("This will run forever")

# Looping from 0 to 4
for i in range(5):
	print(i)
	
# Looping from 2 to 5
for i in range(2, 6):
	print(i)
	
# Looping from 5 to 2 (decrementing by 1)
for i in range(5, 1, -1):
	print(i)
	
# Looping from 31 to 69 with incrementing by 2 each time
for i in range(31, 70, 2):
	print(i)
```

### Math  
The division is decimal by default
```python
print (5 / 2) # 2.5
```
Double slash rounds down, this also includes negative numbers
```python
print (5 // 2) # 2
print (-3 // 2) # -2
```
For round negative numbers towards zero you can use decimal division and then convert to int.
```python
print(int(-3 / 2)) # -1
```
Negative modding might not work as you might expected.
```python
print(-10 % 3) # 2
```
To get -1 from modding negative numbers you can use
```python
import math
print(math.fmod(-10, 3)) # -1.0
```
More math helpers
```python
import math

print(math.floor(3 / 2))
print(math.ceil(3 / 2))
print(math.sqrt(2))
print(math.pow(2, 3))
```
Python numbers are infinite, so they never overflow. If you need to use max/min values you can use
```python
float("inf")
float("-inf")
```
A REALLY HUGE number will always be smaller than `float("inf")`.
### Arrays  
Arrays are called `Lists` in Python. Arrays in Python are dynamic.
```python
arr = [1, 2, 3]
```
Arrays can be used as stacks. Appending and popping a value is an O(1) operation.
```python
arr.append(4)
arr.pop()
```
Also, you can insert a value to a certain index. This is an O(n) operation.
```python
arr.insert(1, 7) # O(n) operation
```
You can set the value of a given index in constant time.
```python
arr[0] = 0 # O(1)
```
A list with a default value can be generated like this
```python
n = 5
arr = [1] * n
```
The index of an array starts from `0`. Indexing `-1` will give you the last value, `-2` will give you the second last value, etc.
```python
arr = [1, 2, 3, 4]
print(arr[0]) # 1
print(arr[-1]) # 4
print(arr[-2]) # 3
```
Sublisting (aka slicing) is a handy tool. Similarly to for-loop ranges, the last index is non-inclusive
```python
arr = [1, 2, 3, 4, 5]
print(arr[1:3]) # [2, 3]
print(arr[0:4]) # [1, 2, 3, 4]
```
Unpacking is also possible but you must be sure that num of variables matches to the array length
```python
a, b, c = [1, 2, 3]
```
As the `range()` function gives us a list, looping another list mechanism is pretty straightforward.
```python
nums = [1, 2, 3]

# with index
for i in range(len(nums)):
	print(nums[i])

# without index
for n in nums:
	print(n)

# with index and value
for i, n in enumerate(nums):
	print(i, n)
```
Looping multiple arrays simultaneously with using `zip()` and unpacking is also possible.
```python
nums1 = [1, 3, 5]
nums2 = [2, 4, 6]

for n1, n2 in zip(nums1, nums2):
	print(n1, n2)
```
Reversing an array is pretty simple
```python
nums.reverse()
```

### Sorting  
Sorting an array is pretty straightforward
```python
arr.sort()
```
To sort in descending order you can set reverse argument `True`
```python
arr.sort(reverse=True)
```
Strings will be sorted in alphabetical order
```python
arr = ["bob", "alice", "jane", "doe"]
arr.sort() # ["alice", "bob", "doe", "jane"]
```
If you need custom sorting (sorting by length for example) you can set `key` argument to a `lambda function`. 
> For this case our lambda function means: for every value 'x', map it with len(x).
```python
arr.sort(key=lambda x:len(x))
```

### List Comprehension  
Creating lists with initial values can easily be done by using list comprehensions.
```python
arr = [i for i in range(5)] # [0, 1, 2, 3, 4]
arr = [i+i for i in range(5)] # [0, 2, 4, 6, 8]
```

### 2D Arrays  
You can easily create a 2D array with list comprehension
```python
arr = [[0] * 4 for i in range(4)]
```
If you think this can be simplified by multiplying a list, unfortunately, it won't work.
```python
# This won't work
# arr = [[0] * 4] * 4
```

### Strings  
Strings can be generated by using `'` and `"`.
```python
s = "abc"
ss = 'abc'
```
You can get a substring with slicing like arrays
```python
print(s[0:2]) #ab
```
Strings are immutable, so you can't change a char in a certain index. Modifying a string will create a new string.
```python
# This won't work
# s[0] = "A"

# This creates a new string
s += "def" # O(n)
```
If you need the ASCII value of a char
```python
print(ord("a")) # 97
```
You can combine a list of strings with `.join()`
```python
strings = ["ab", "cd", "ef"]
print("".join(strings)) # abcdef
print("-".join(strings)) # ab-cd-ef
```

### Queues  
In Python, queues are double-ended by default
```python
from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
print(queue) # deque([1, 2])

queue.popleft()
print(queue)  # deque([2]) # O(1)

queue.appendleft(1)
print(queue) # deque([1, 2])

queue.pop()
print(queue) # deque([1])
```

### Hash Sets  
We can search and insert in O(1) in Hash Sets.
```python
mySet = set()

mySet.add(1)
mySet.add(2)
print(mySet) # {1, 2}
print(len(mySet)) # 2
print(1 in mySet) # True
print(2 in mySet) # True
print(3 in mySet) # False

mySet.remove(2)
print(2 in mySet) # False
```
Sets can be generated from lists. And also Set comprehensions.
```python
# Generate from list
print(set([1, 2, 3])) # {1, 2, 3}

# Generate with set comprehension
mySet = { i for i in range(5) }
```

### Hash Maps  
Hash Maps (a.k.a dictionaries) is one of the most used data collections for coding interviews. Creating a hash map is easy. You can assign or update a `key` to a `value`. For example, string `alice` is the key here and 88 is the value of the key.
```python
myMap = {}
myMap["alice"] = 88
myMap["bob"] = 77

#or
myMap = {"alice": 88, "bob": 77}

# We can change a value of key
myMap["alice"] = 80
```
We can look for if a key exists or not
```python
print("alice" in myMap) # True
myMap.pop("alice")
print("alice" in myMap) # False
```
Like list comprehension, also using dict comprehensions is possible.
```python
myMap = { i: 2*i for i in range(3)}
```
Looping through maps is also pretty useful. The code is pretty self-exploratory
```python
for key in myMap:
	print(key, myMap[key])

for val in myMap.values():
	print(val)

for key, val in myMap.items():
	print(key, val)
```
For some cases, creating the dict with defaultdict prevents a lot of edge case crashes
```python
from collections import defaultdict

# Initialize a defaultdict with int, which defaults missing keys to 0
counts = defaultdict(int)

# Count occurrences in a list
for item in ["apple", "banana", "apple"]:
    counts[item] += 1
```
### Tuples  
Tuples are like arrays but immutable
```python
tup = (1, 2, 3)
print(tup)
print(tup[-1])
```
Tuples also can be used as a key for hash map/set But lists can't be keys
```python
myMap = { (1,2): 3 }
print(myMap[(1,2)])

mySet = set()
mySet.add((1, 2))
print((1, 2) in mySet) # True
```
Unpacking is also possible for tuples
```python
fruits = ("apple", "banana", "cherry")  
  
(green, yellow, red) = fruits  
  
print(green) # apple
print(yellow) # banana
print(red) # cheery
```
### Heaps  
Heaps might be useful for some kind of questions. Python heaps are min heaps by default. Under the hood, heaps are arrays
```python
import heapq

minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)

# Min is always at index 0
print(minHeap[0]) # 2

while len(minHeap):
	print(heapq.heappop(minHeap))
# 2 
# 3
# 4
```
There is no max heaps by default. To work around this, you can use min heap but multiply the values by -1 when doing push & pop operations
```python
maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)

# Max is always at index 0
print(-1 * maxHeap[0])

while len(minHeap):
	print(-1 * heapq.heappop(minHeap))
# 4
# 3
# 2
```
Heaps can also be built with initial values.
```python
arr = [2, 1, 8, 4, 5]
heapq.heapify(arr) # O(n)
```

### Functions  
```python
def myFunc(n, m):
	return n * m

print(myFunc(3, 4))
```

### Nested Functions  
Nested functions have access to outer variables
```python
def outer(a, b):
	c = "c"

	def inner():
		return a + b + c
	return inner()

print(outer("a", "b")) # abc
```
They also can modify objects but not reassign them unless `nonlocal` keyword is used
```python
def double(arr, val):
	def helper():
		# Modifying array works
		for i, n in enumerate(arr):
			arr[i] *= 2

		# will only modify val in the helper scope
		# val *= 2

		# this will modify val outside helper scope
		nonlocal val
		val *= 2
	helper()
	print(arr, val)

nums = [1, 2]
val = 3
double(nums, val) # [2, 4] 6
```

### Classes
Ofc, there is a lot more about Python classes. However, knowing `__init__` and usage of `self` should be enough for interviews.
```python
class MyClass:
	# Constructor
	def __init__(self, nums):
		self.nums = nums
		self.size = len(nums)

	# self keyword required as param
	def getLength(self):
		return self.size

	def getDoubleLength(self):
		return 2 * self.getLength()
```

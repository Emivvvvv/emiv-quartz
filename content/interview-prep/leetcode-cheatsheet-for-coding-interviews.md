---
title: LeetCode Cheatsheet for Coding Interviews
draft: false
tags:
  - Algorithms
  - DataStructures
  - Python
  - Interview
---
# Intro

As someone preparing for interviews daily, I thought a cheat sheet focused on common patterns I might be asked about would be beneficial. Here, you’ll find the most common patterns, along with explanations, code templates I use to solve these questions, and an example question that I found both interesting and helpful in understanding each pattern better. I use Python for my interviews, so if you do too—or want to start using it—I recommend checking out my [[python-for-interviews-cheatsheet|Python for Interviews Cheatsheet]].

## Arrays & Hashing
---
### Prefix Sum

#### Idea

Prefix Sum is a technique used to store cumulative sums of a sequence of numbers, allowing for efficient range sum calculations. By precomputing the sum of elements up to each index, you can quickly calculate the sum of any subarray.

#### Template

```python
def prefix_sum_template(arr):
    prefix_sum = [0] * (len(arr) + 1)
    
    # Compute prefix sum
    for i in range(1, len(arr) + 1):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]
    
    # Example use: Getting sum of range [i, j]
    def range_sum(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]
```

#### Example Question: [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

**Example:**

> **Input:** nums = [1,2,3,4]<br>
> **Output:** [24,12,8,6]<br>
> **Explanation:** For each index, the product of all elements except the current one results in the array [24,12,8,6].

Solution:

```python
def productExceptSelf(self, nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n
    
    # Calculate prefix products
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    
    # Calculate suffix products and multiply
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    
    return res
```

---
### Two Pointers

#### Idea

The Two Pointers technique uses two pointers to traverse a data structure, typically one starting from the beginning and the other from the end. 

#### Template

```python
def two_pointers_template(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Check a condition based on both pointers
        if some_condition(arr[left], arr[right]):
            # Perform some action or store result
            process(arr[left], arr[right])
        
        # Adjust pointers based on specific requirements
        if adjust_left_condition:
            left += 1
        else:
            right -= 1
```

#### Example Question: [15. 3Sum](https://leetcode.com/problems/3sum/)

Given an integer array `nums`, return all the unique triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

**Example:**

> **Input:** nums = [-1, 0, 1, 2, -1, -4]  <br>
> **Output:** [[-1, -1, 2], [-1, 0, 1]]  <br>
> **Explanation:** The unique triplets that sum to zero are [-1, -1, 2] and [-1, 0, 1].

Solution:

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()  # Sort the array
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                
                while left < right and nums[left] == nums[left - 1]:
                    left += 1

            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

---
### Sliding Window

#### Idea

The Sliding Window technique is used to traverse subsets of a data structure, often an array or string, by maintaining a "window" of elements and sliding it across the structure. This approach is effective for finding subarrays or substrings that meet a specific condition.

#### Template (fixed length window)

```python
def fixed_sliding_window_template(arr, k):
    window_sum = 0
    max_sum = float('-inf')
    
    # Initialize the first window
    for i in range(k):
        window_sum += arr[i]
    
    max_sum = max(max_sum, window_sum)
    
    # Slide the window, starting from the k-th element
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

Template (dynamic length window)

```python
def dynamic_sliding_window_template(arr, target):
    window_sum = 0
    min_length = float('inf')
    start = 0
    
    for end in range(len(arr)):
        window_sum += arr[end]
        
        # Shrink the window as small as possible
        # while the window_sum is >= target
        while window_sum >= target:
            min_length = min(min_length, end - start + 1)
            window_sum -= arr[start]
            start += 1
            
    return min_length if min_length != float('inf') else 0
```

#### Example Question: [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

Given a string `s` and an integer `k`, you can choose any character in the string and change it to any other uppercase character at most `k` times. Return the length of the longest substring containing the same letter you can get after performing the above operations.

**Example:**

> **Input:** s = "AABABBA", k = 1  <br>
> **Output:** 4  <br>
> **Explanation:** Replace the one 'B' in "AABABB" to get "AAAA" with a length of 4.

Solution:

```python
def characterReplacement(self, s: str, k: int) -> int:
    left = 0
    max_count = 0
    counts = {}
    max_length = 0
    
    for right in range(len(s)):
        counts[s[right]] = counts.get(s[right], 0) + 1
        max_count = max(max_count, counts[s[right]])
        
        # If the window size - max_count exceeds k, shrink the window
        if (right - left + 1) - max_count > k:
            counts[s[left]] -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

---
### Intervals

#### Idea

Interval problems involve processing a set of intervals, typically with operations like merging, finding overlaps, or checking if intervals are covered. Sorting intervals and using a greedy approach or maintaining a count of overlapping intervals is common in these problems.
#### Example Question: [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

Given an array of intervals where intervals[i] = `[start, end]`, merge all overlapping intervals, and return an array of the non-overlapping intervals.

**Example:**

> **Input:** intervals = \[[1,3],[2,6],[8,10],[15,18]]<br>
> **Output:** \[[1,6],[8,10],[15,18]]

Solution:

```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = []
    
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged
```

---
### Hashmaps

#### Idea

Hashmaps, or dictionaries, are collections that map keys to values, allowing for efficient O(1) average-time complexity for lookups, insertions, and deletions. They are especially useful for problems that require fast access to values associated with unique keys.

#### Example Question: [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

Given an array of strings `strs`, group the anagrams together.

**Example:**

> **Input:** strs = ["eat","tea","tan","ate","nat","bat"]  <br>
> **Output:** \[["bat"],["nat","tan"],["ate","eat","tea"]]

Solution:

```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    anagrams = {}
    
    for s in strs:
        # Sort characters as key
        key = "".join(sorted(s))
        if key in anagrams:
            anagrams[key].append(s)
        else:
            anagrams[key] = [s]
    
    return list(anagrams.values())
```

---
### Sets

#### Idea

Sets are unordered collections of unique elements, providing efficient O(1) average-time complexity for element lookups, insertions, and deletions. For problems involving unique elements or distinct sequences, sets can simplify solutions.
#### Example Question: [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

Given an unsorted array of integers `nums`, find the length of the longest consecutive elements sequence.

**Example:**

> **Input:** nums = [100,4,200,1,3,2]  <br>
> **Output:** 4  <br>
> **Explanation:** The longest consecutive elements sequence is `[1, 2, 3, 4]`.

Solution:

```python
def longestConsecutive(self, nums: List[int]) -> int:
    num_set = set(nums)
    longest_streak = 0
    
    for num in num_set:
        if num - 1 not in num_set:  # Start of a new sequence
            current_num = num
            current_streak = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1
            
            longest_streak = max(longest_streak, current_streak)
    
    return longest_streak
```

---
## Stack

#### Idea

A stack is a data structure that follows the Last In, First Out (LIFO) principle. It’s useful for scenarios that involve reversing elements, tracking function calls, or balancing symbols (e.g., parentheses). Common operations include pushing elements onto the stack, popping elements off, and checking the top element.
#### Example Question: [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

Given a list of daily temperatures `temperatures`, return a list such that, for each day in the input, it tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put `0` instead.

**Example:**

> **Input:** temperatures = [73, 74, 75, 71, 69, 72, 76, 73]  <br>
> **Output:** [1, 1, 4, 2, 1, 1, 0, 0]  <br>
> **Explanation:** For each day, you wait until a warmer temperature appears.

Solution:

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result = [0] * len(temperatures)
        stack = []
        
        for i, temp in enumerate(temperatures):
	        # Access temperature in the tuple
            while stack and stack[-1][1] < temp:  
                # Unpack the tuple (index, temperature)
                index, _ = stack.pop()  
                result[index] = i - index
            stack.append((i, temp))  # Push tuple (index, temperature)
        
        return result
```

---
## Binary Search

#### Idea

Binary Search is an efficient algorithm for finding a target value in a sorted array by repeatedly dividing the search interval in half. It eliminates half of the remaining elements each step, making it ideal for large datasets. This technique requires the data to be sorted beforehand.

#### Template

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] > target:
	        right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            return mid  # Target found
    
    return -1  # Target not found
```

#### Example Question: [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

Write an efficient algorithm that searches for a value in an `m x n` matrix. Integers in each row are sorted from left to right, and the first integer of each row is greater than the last integer of the previous row.

**Example:**

> **Input:** matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], target = 3  <br>
> **Output:** true  

Solution:

```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // cols][mid % cols]
        
        if mid_value < target:
            left = mid + 1
        elif mid_value > target:
            right = mid - 1
		else:
            return True

    return False
```

This solution uses **binary search** by treating the 2D matrix as a 1D array, enabling efficient searching in O(log(m \* n)) time.
#### Example Question: [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

Given a rotated sorted array `nums` of unique elements, find the minimum element.

**Example:**

> **Input:** nums = [3,4,5,1,2]  <br>
> **Output:** 1  

Solution:

```python
def findMin(self, nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]
```

---
## Linked List

#### Idea

A linked list is a data structure made up of nodes, where each node contains a value and a reference (or pointer) to the next node. Linked lists are ideal for dynamic data structures where elements are frequently inserted or deleted, as they allow for efficient O(1) insertion and deletion.

#### Template

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def linked_list_template(head):
    curr = head
    while curr:
        # Perform an action on the current node
        process(curr)
        curr = curr.next
```

#### Example Question: [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

Given the head of a singly linked list, reverse the list, and return the reversed list.

**Example:**

> **Input:** head = [1,2,3,4,5]  <br>
> **Output:** [5,4,3,2,1]  

Solution:

```python
def reverseList(self, head: ListNode) -> ListNode:
    prev = None
    curr = head
    
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    return prev
```

---
## Tree

A tree is a hierarchical data structure made up of nodes, where each node has a value and references to its child nodes.  Traversal methods (in-order, pre-order, post-order) are used to visit nodes in a specific order.

### Binary Tree
#### Idea

A binary tree has nodes with up to two children: left and right. Common types include full (0 or 2 children per node), complete (all levels filled except possibly the last), perfect (all interior nodes have two children and all leaves have the same depth)
#### Template

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_traversal_template(root):
    if not root:
        return
    
    # Pre-order traversal
    process(root)  # Process node before its children
    tree_traversal_template(root.left)
    tree_traversal_template(root.right)

    # In-order traversal
    tree_traversal_template(root.left)
    process(root)  # Process node between children
    tree_traversal_template(root.right)

    # Post-order traversal
    tree_traversal_template(root.left)
    tree_traversal_template(root.right)
    process(root)  # Process node after its children
```

#### Example Question: [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

Given the `root` of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

**Example:**

> **Input:** root = [1,2,2,3,4,4,3]  <br>
> **Output:** true  

Solution:

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val) and is_mirror(left.left, right.right) and is_mirror(left.right, right.left)
    
    return is_mirror(root.left, root.right)
```
#### Example Question: [112. Path Sum](https://leetcode.com/problems/path-sum/)

Given the `root` of a binary tree and an integer `targetSum`, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals `targetSum`.

**Example:**

> **Input:** root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22  
> **Output:** true  

Solution:

```python
def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    targetSum -= root.val
    return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
```

#### Example Question: [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

Given the `root` of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).

**Example:**


>Input: root = [3,9,20,null,null,15,7]<br>
>Output: [[3],[9,20],[15,7]]


**Solution:**

```python
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result
```
---
### Binary Search Tree

#### Idea

A Binary Search Tree (BST) is a tree where each node has at most two children. For each node, the left subtree contains only nodes with values less than the node's value, and the right subtree contains only nodes with values greater than the node's value. In-order traversal gives the values sorted in BST.

#### Template

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def bst_insert(root, value):
    if not root:
        return TreeNode(value)
        
    elif value < root.val:
        root.left = bst_insert(root.left, value)
    else:
        root.right = bst_insert(root.right, value)
    return root
```

#### Example Question: [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes `p` and `q`.

**Example:**

> **Input:** root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8  <br>
> **Output:** 6  <br>
> **Explanation:** The lowest common ancestor of nodes 2 and 8 is 6.

Solution:

```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    current = root
    
    while current:
        if p.val < current.val and q.val < current.val:
            current = current.left
        elif p.val > current.val and q.val > current.val:
            current = current.right
        else:
            return current
```

#### Example Question: [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

Given the root of a binary search tree and an integer `k`, return the `k`th smallest element in the BST.

**Example:**

> **Input:** root = [3,1,4,null,2], k = 1  <br>
> **Output:** 1  

Solution:

```python
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
	arr = []
	def inorder(node):
		if not node:
			return 0

		inorder(node.left)
		arr.append(node.val)
		inorder(node.right)
		
	inorder(root)
	return arr[k - 1]
```

---

## Heap / Priority Queue

#### Idea

A heap is a specialized binary tree that satisfies the heap property, where the parent node is either greater than (max-heap) or less than (min-heap) its children. Heaps are often used to implement priority queues, where elements are removed based on their priority. Heaps are efficient for operations that need the maximum or minimum element frequently.

#### Template

```python
import heapq

def heap_operations(elements):
    # Min-heap
    min_heap = []
    for el in elements:
        heapq.heappush(min_heap, el)
    
    # Access and remove the smallest element
    min_element = heapq.heappop(min_heap)
    
    return min_element
```

#### Example Question: [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

Given an integer array `nums` and an integer `k`, return the `k`th largest element in the array.

**Example:**

> **Input:** nums = [3,2,1,5,6,4], k = 2  <br>
> **Output:** 5  

Solution:

```python
def findKthLargest(self, nums: List[int], k: int) -> int:
	# since we don't have max heap in python we use this trick
	# to use min heap as max heap.
    nums= [-x for x in nums]
	heapq.heapify(nums)

	kth = None
	for i in range(k):
		kth = heapq.heappop(nums)

	# don't forget to restore the value
	return -kth
```

---
## Backtracking

#### Idea

Backtracking builds solutions step-by-step, exploring each option. If a choice doesn’t work, it “backtracks” by undoing it and trying the next option. This approach is ideal for exploring combinations, permutations, and puzzle solutions.

#### Template

```python
def backtrack_template(parameters):
    res = []

    def backtrack(path = []):
        # Base case: Check if the current path 
        # meets the target condition
        if is_target_met(path):
	        # Add a copy of the current path to res
            res.append(path[:]) 
            return

        # Recursive case: Iterate through all choices
        for choice in parameters:
            # Check if adding the choice maintains the constraints
            if is_valid_choice(path, choice):
                # Recursive call with the choice added to the path
                backtrack(path + [choice])

    # Initial call to the backtrack function
    backtrack()
    return res
```


#### Example Question: [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

Given `n` pairs of parentheses, write a function to _generate all combinations of well-formed parentheses_.

**Example 1:**

> **Input:** n = 3<br>
> **Output:** ["((()))","(()())","(())()","()(())","()()()"]

**Example 2:**

> **Input:** n = 1<br>
> **Output:** ["()"]

Solution:

```python
def generateParenthesis(self, n: int) -> List[str]:
	res = []
	
	def backtrack(open = 0, close = 0, curr = []):
		if n*2 == len(curr):
			res.append("".join(curr))
			return
			
		if open < n:
			backtrack(open + 1, close, curr + ["("])
		if open > close:
			backtrack(open, close + 1, curr + [")"])
	
	backtrack()
	return res
```

## Graph and Dynamic Programming

I wanted to review [[graphs|Graphs]] and [[dynamic-programming|Dynamic Programming]] more comprehensively since they are combinations of many topics we've discussed. So you can check them out from their posts.
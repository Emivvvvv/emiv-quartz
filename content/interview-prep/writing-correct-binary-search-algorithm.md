---
title: Writing Correct Binary Search Algorithm
draft: false
tags:
  - Algorithms
  - Interview
  - Python
---
### **TL;DR**

1. Is the problem finding a specific value or a boundary?
    - Specific value: Use `left <= right`.
    - Boundary: Use `left < right`.
2. Does `mid` need to be included as a possible answer?
    - Yes: Use `left = mid` or `right = mid`.
    - No: Use `left = mid + 1` or `right = mid - 1`.

---
### **Intro**

If you're given a sorted list or they request an algorithm that has TC O(logn), your to-go algorithm should generally be **Binary Search**. Implementing a binary search is actually easy, but the loop condition and assigning the left and right values might differ from problem to problem. In this post, I tried to understand and explain what to use in which case.

---

### **Loop Condition: `left < right` vs. `left <= right`**

#### **`left < right`**

- Use this when the termination condition is when the search space is reduced to a **single element**.
- After the loop, `left` will be the index of the result.

**Example**: Find the minimum element in a rotated sorted array.

- The loop continues until `left == right`, and you return `nums[left]`.

#### **`left <= right`**

- Use this when you want to check **all elements** in the search space, including the possibility of `left == right` at some point in the loop.
- Typically used when the loop checks for equality or an exact match (e.g., finding a target value).

**Example**: Search for a specific element in a sorted array.

- The loop exits when `left > right`, meaning the target is not present.

### **Assignments: `left = mid + 1` vs. `left = mid`**

#### **`left = mid + 1`**

- Use this when you are certain that `mid` is **not the answer** and the next possible candidate is to the right of `mid`.

**Example**:

- When the condition specifies the target must be in the right half (e.g., `nums[mid] < target` in standard binary search).

#### **`left = mid`**

- Use this when `mid` could still be the answer.
- Often used when the problem involves finding a boundary or minimum/maximum.

**Example**:

- When looking for the minimum in a rotated array, the `mid` itself might be the minimum, so you don’t exclude it.

---

### **Assignments: `right = mid - 1` vs. `right = mid`**

#### **`right = mid - 1`**

- Use this when you are certain that `mid` is **not the answer**, and the next possible candidate is to the left of `mid`.

**Example**:

- When the condition specifies the target must be in the left half (e.g., `nums[mid] > target`).

#### **`right = mid`**

- Use this when `mid` could still be the answer.
- Common in problems where you need to find the minimum or maximum.

**Example**:

- Finding the boundary or the smallest/largest value in a rotated array.

---
### **Examples**

#### Example 1: Search for a Target in a Sorted Array

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:  # Check all elements
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1  # Exclude mid, go right
        else:
            right = mid - 1  # Exclude mid, go left
    return -1
```

#### Example 2: Find the Minimum in a Rotated Array

```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:  # Narrow to a single element
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1  # Minimum is in the right half
        else:
            right = mid  # Include mid, minimum could be here
    return nums[left]
```

#### Example 3: First Bad Version

```python
def firstBadVersion(n):
    left, right = 1, n
    while left < right:  # Narrow to a single element
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid  # Include mid, could be the first bad version
        else:
            left = mid + 1  # Exclude mid, go right
    return left
```

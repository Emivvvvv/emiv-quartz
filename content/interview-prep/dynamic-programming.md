---
title: Dynamic Programming
draft: true
tags:
---
# Intro
### ****1. Top-Down Approach (Memoization):****

In the top-down approach, also known as ****memoization****, we start with the final solution and recursively break it down into smaller subproblems. To avoid redundant calculations, we store the results of solved subproblems in a memoization table.

### ****2. Bottom-Up Approach (Tabulation):****

In the bottom-up approach, also known as ****tabulation****, we start with the smallest subproblems and gradually build up to the final solution. We store the results of solved subproblems in a table to avoid redundant calculations.

## Group 1

Basic questions to get a feel of DP.
[https://leetcode.com/problems/climbing-stairs/](https://leetcode.com/problems/climbing-stairs/)
## Group 2 (linear sequence, linear time, constant transition):

Dp solution requires us to solve the sub problem on every prefix of the array. A prefix of the array is a subarray from 0 to i for some i.
[https://leetcode.com/problems/min-cost-climbing-stairs/](https://leetcode.com/problems/min-cost-climbing-stairs/)

## Group 3 (on grids):

Dp table will have the same dimensions as grid, the state at cell i,j will be related to the grid at cell i,j.

[https://leetcode.com/problems/unique-paths/](https://leetcode.com/problems/unique-paths/)
## Group 4 (two sequences, O(NM) style):

Dp[i][j] is some value related to the problem solved on prefix of sequence 1 with length i, and prefix on sequence 2 with length j.

[https://leetcode.com/problems/longest-common-subsequence/](https://leetcode.com/problems/longest-common-subsequence/)
## Group 5 (Interval DP):

Dp problem is solved on every single interval (subarray) of the array

[https://leetcode.com/problems/longest-palindromic-subsequence/](https://leetcode.com/problems/longest-palindromic-subsequence/)

## Group 6 (linear sequence transition like N2 Longest Increasing Subsequence)

Dp problem is solved on every prefix of the array. Transition is from every index j < i.

[https://leetcode.com/problems/partition-array-for-maximum-sum/](https://leetcode.com/problems/partition-array-for-maximum-sum/)

## Group 7 (knapsack-like)

Dp state is similar to the classical knapsack problem.

[https://leetcode.com/problems/partition-equal-subset-sum/](https://leetcode.com/problems/partition-equal-subset-sum/)

# References

- https://www.reddit.com/r/leetcode/comments/14o10jd/the_ultimate_dynamic_programming_roadmap/
- https://www.youtube.com/watch?v=9k31KcQmS_U
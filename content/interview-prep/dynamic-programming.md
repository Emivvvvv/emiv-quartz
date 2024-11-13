---
title: Dynamic Programming
draft: true
tags:
  - Algorithms
  - DataStructures
  - Python
  - Interview
---
---
## Minimum (Maximum) Path to Reach a Target

### Overview

This pattern involves finding the minimum or maximum cost/path to reach a target. Problems in this category often require you to make decisions at each step that contribute to an overall optimal solution.

### Key Characteristics

- **Optimization Objective**: Minimize or maximize a certain value.
- **Subproblem Overlapping**: The problem can be broken down into smaller, overlapping subproblems.
- **State Definition**: The DP state typically represents the minimum or maximum value to reach a certain point.

### Example Problem: [322. Coin Change](https://leetcode.com/problems/coin-change/)

#### Problem Description

Given an integer array `coins` representing coins of different denominations and an integer `amount`, return the fewest number of coins needed to make up that amount. If it's impossible, return `-1`.

#### Solution Explanation

We need to find the minimum number of coins that sum up to the target amount. This is a classic example of the minimum path to reach a target.

##### Steps

1. **Define the DP Array**: `dp[i]` represents the minimum number of coins needed to make up amount `i`.

2. **Initialize the DP Array**: Set `dp[0] = 0` since zero coins are needed to make up amount zero. Initialize the rest with a value greater than the maximum possible (e.g., `amount + 1`).

3. **Iterate and Update**: For each amount from `1` to `amount`, iterate through the coins and update `dp[i]` accordingly.

4. **Return the Result**: If `dp[amount]` is greater than `amount`, return `-1`, indicating it's impossible to make up the amount. Otherwise, return `dp[amount]`.

##### Code

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize DP array
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        # Build up the DP array
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        # Check if the amount can be made up
        return dp[amount] if dp[amount] <= amount else -1
```

##### Complexity Analysis

- **Time Complexity**: \( O(amount \times n) \), where \( n \) is the number of coin denominations.
- **Space Complexity**: \( O(amount) \), for the DP array.

---

## Distinct Ways

### Overview

This pattern deals with counting the number of distinct ways to achieve a certain goal. It's common in problems where you need to find the number of different combinations or sequences that satisfy certain conditions.

### Key Characteristics

- **Counting Solutions**: Focused on counting the number of ways rather than finding the way itself.
- **Recursive Substructure**: The total number of ways can be expressed in terms of the number of ways to reach smaller subproblems.
- **State Definition**: The DP state usually represents the number of ways to reach a particular state or sum.

### Example Problem: [494. Target Sum](https://leetcode.com/problems/target-sum/)

#### Problem Description

Given an integer array `nums` and an integer `target`, you can add `'+'` or `'-'` before each number in `nums` to form an expression. Return the number of different expressions that evaluate to `target`.

#### Solution Explanation

We need to find the number of ways to assign `'+'` or `'-'` to each number such that the total sum equals the target.

##### Steps

1. **Calculate the Total Sum**: Compute the sum of all numbers in `nums`.

2. **Check Feasibility**: If `(sum_nums + target)` is odd or `target` is greater than `sum_nums`, return `0`.

3. **Transform the Problem**: Convert it into a subset sum problem:
   \[
   \text{Positive subset sum} = \frac{\text{sum\_nums} + \text{target}}{2}
   \]

4. **Define the DP Array**: `dp[i]` represents the number of ways to reach sum `i`.

5. **Initialize the DP Array**: `dp[0] = 1`, as there's one way to reach sum `0` (by choosing no elements).

6. **Iterate and Update**: For each number in `nums`, update the DP array in reverse to avoid overwriting.

##### Code

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sum_nums = sum(nums)
        if (sum_nums + target) % 2 != 0 or target > sum_nums:
            return 0

        s = (sum_nums + target) // 2
        dp = [0] * (s + 1)
        dp[0] = 1

        for num in nums:
            for i in range(s, num - 1, -1):
                dp[i] += dp[i - num]

        return dp[s]
```

##### Complexity Analysis

- **Time Complexity**: \( O(n \times s) \), where \( s \) is the subset sum.
- **Space Complexity**: \( O(s) \), for the DP array.

---

## Merging Intervals

### Overview

This pattern is used in problems where you need to combine or partition sequences or intervals to minimize or maximize a cost.

### Key Characteristics

- **Non-overlapping Intervals**: Problems often involve merging or partitioning intervals without overlap.
- **State Definition**: The DP state represents the minimum or maximum cost to merge a range of intervals.
- **Decision Making**: At each step, decide where to partition the sequence.

### Example Problem: [1000. Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/)

#### Problem Description

Given an array `stones`, merge them into piles of size `K`. The cost of merging adjacent piles is equal to the sum of their total stones. Return the minimum cost to merge all piles into one pile. If it's impossible, return `-1`.

#### Solution Explanation

We need to find the minimum cost to merge the stones according to the rules. This is a variation of the interval DP problem.

##### Steps

1. **Check Feasibility**: If `(len(stones) - 1) % (K - 1)` is not zero, it's impossible to merge into one pile.

2. **Compute Prefix Sums**: Use prefix sums to calculate the sum of stones in a range efficiently.

3. **Define the DP Array**: `dp[i][j]` represents the minimum cost to merge stones from index `i` to `j`.

4. **Initialize the DP Array**: Set all values to `0` where `i == j`.

5. **Iterate and Update**:
   - For all ranges of length from `K` to `n`:
     - Try all possible partitions and update `dp[i][j]`.
     - If the current range can form one pile, add the total sum to `dp[i][j]`.

##### Code

```python
class Solution:
    def mergeStones(self, stones: List[int], K: int) -> int:
        n = len(stones)
        if (n - 1) % (K - 1):
            return -1

        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + stones[i]

        dp = [[0] * n for _ in range(n)]
        for m in range(K, n + 1):
            for i in range(n - m + 1):
                j = i + m - 1
                dp[i][j] = min(dp[i][t] + dp[t + 1][j] for t in range(i, j, K - 1))
                if (m - 1) % (K - 1) == 0:
                    dp[i][j] += prefix[j + 1] - prefix[i]

        return dp[0][n - 1]
```

##### Complexity Analysis

- **Time Complexity**: \( O(n^3 / K) \), due to the nested loops.
- **Space Complexity**: \( O(n^2) \), for the DP array.

---

## DP on Strings

### Overview

This pattern involves solving problems related to string manipulation, such as finding subsequences, substrings, or matching patterns.

### Key Characteristics

- **Two-Dimensional DP**: Often requires a 2D DP array since operations involve two strings or two indices.
- **State Definition**: The DP state typically represents the solution to a substring or subsequence ending at certain indices.
- **Common Problems**: Longest Common Subsequence, Edit Distance, String Matching.

### Example Problem: [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

#### Problem Description

Given two strings `text1` and `text2`, return the length of their longest common subsequence. A subsequence is a sequence that appears in the same relative order but not necessarily contiguous.

#### Solution Explanation

We need to find the length of the longest sequence that appears in both strings.

##### Steps

1. **Define the DP Array**: `dp[i][j]` represents the length of the longest common subsequence between `text1[0...i-1]` and `text2[0...j-1]`.

2. **Initialize the DP Array**: `dp[0][j] = 0` and `dp[i][0] = 0` because an empty string has a common subsequence length of `0`.

3. **Iterate and Update**:
   - If `text1[i-1] == text2[j-1]`, then `dp[i][j] = dp[i-1][j-1] + 1`.
   - Else, `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

4. **Return the Result**: `dp[m][n]` will have the length of the longest common subsequence.

##### Code

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        # Initialize DP array
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Build up the DP array
        for i in range(m):
            for j in range(n):
                if text1[i] == text2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        # The bottom-right cell contains the answer
        return dp[m][n]
```

##### Complexity Analysis

- **Time Complexity**: \( O(m \times n) \), where \( m \) and \( n \) are the lengths of the two strings.
- **Space Complexity**: \( O(m \times n) \), for the DP array.

---

## Decision Making

### Overview

This pattern involves making decisions at each step to optimize a certain objective, often under certain constraints.

### Key Characteristics

- **Choice at Each Step**: Decide whether to take an action or skip it.
- **State Definition**: The DP state includes parameters that represent decisions made so far.
- **Common Problems**: Stock trading, scheduling with penalties, or any scenario where decisions affect future outcomes.

### Example Problem: [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

#### Problem Description

You are given an array `prices` where `prices[i]` is the price of a stock on day `i` and an integer `fee` representing a transaction fee. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction. Return the maximum profit you can achieve.

#### Solution Explanation

We need to maximize profit by deciding whether to buy, sell, or do nothing on each day.

##### Steps

1. **Define the DP States**:
   - `hold`: Maximum profit when holding a stock.
   - `cash`: Maximum profit when not holding a stock.

2. **Initialize the States**:
   - `hold = -prices[0]` (We buy the first stock).
   - `cash = 0` (No profit initially).

3. **Iterate and Update**:
   - For each price in `prices[1:]`:
     - Update `cash` as the maximum of itself and `hold + price - fee` (sell stock).
     - Update `hold` as the maximum of itself and `cash - price` (buy stock).

4. **Return the Result**: The `cash` variable will have the maximum profit.

##### Code

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        hold = -prices[0]
        cash = 0

        for price in prices[1:]:
            cash = max(cash, hold + price - fee)
            hold = max(hold, cash - price)

        return cash
```

##### Complexity Analysis

- **Time Complexity**: \( O(n) \), where \( n \) is the number of days.
- **Space Complexity**: \( O(1) \), as we use constant extra space.
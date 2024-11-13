---
title: Graphs
draft: false
tags:
---
# Intro

Graphs are one of the most frequently asked topics in interviews because they can encapsulate multiple concepts in a single question. To be good at graph problems, it’s essential to have strong fundamentals. After I started covering the LeetCode patterns I believe are important to know in the [[leetcode-cheatsheet-for-coding-interviews|LeetCode Cheatsheet for Coding Interviews]] post, now it's time to focus on graph problems and patterns. Given their importance, I decided to dedicate a separate post to graphs. As mentioned in my previous blog, I use Python for my interviews, so if you do too—or are considering it—I recommend checking out my [[python-for-interviews-cheatsheet|Python for Interviews Cheatsheet]].
## Graph Representation

Graph representation is crucial as it affects the efficiency of graph algorithms. The two primary ways to represent graphs are:

- **Adjacency Matrix**: A 2D array where each cell `(i, j)` indicates the presence (and possibly weight) of an edge between vertices `i` and `j`.
  - **Pros**: Quick edge lookup (`O(1)` time).
  - **Cons**: Consumes `O(V^2)` space, inefficient for sparse graphs.
  
- **Adjacency List**: An array of lists where each index represents a vertex, and each element in the list represents its adjacent vertices.
  - **Pros**: Space-efficient for sparse graphs (`O(V + E)` space).
  - **Cons**: Edge lookup can take `O(V)` time in the worst case.
#### Template

**Adjacency List in Python:**

```python
# For an undirected graph
graph = [[] for _ in range(num_vertices)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# For a directed graph
graph = [[] for _ in range(num_vertices)]
for u, v in edges:
    graph[u].append(v)
```

also, creating a dictionary that holds lists is a possible implementation

```python
from collections import defaultdict

# For an undirected graph
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# For a directed graph
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
```

**Adjacency Matrix in Python:**

```python
# For an undirected graph
graph = [[0] * num_vertices for _ in range(num_vertices)]
for u, v in edges:
    graph[u][v] = 1
    graph[v][u] = 1

# For a directed graph
graph = [[0] * num_vertices for _ in range(num_vertices)]
for u, v in edges:
    graph[u][v] = 1
```

---

## Graph Traversal

### Depth-First Search (DFS)

#### Idea

Depth-First Search (DFS) explores as far as possible along each branch before backtracking. It uses a stack data structure, either implicitly with recursion or explicitly, to keep track of the vertices to be explored next. DFS is useful for:

- Detecting cycles
- Topological sorting
- Connectivity checks

#### Template

```python
visited = set()

def dfs(node):
    visited.add(node)
    # Process the current node here if needed
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor)
```

#### Example Problem: [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

**Problem Description:**

Given an `m x n` matrix of non-negative integers representing the height of each unit cell in a continent, find the list of grid coordinates where water can flow to both the Pacific and Atlantic oceans. Water can flow from a cell to its neighboring cells (up, down, left, or right) if the neighboring cell's height is less than or equal to the current cell's height.

**Solution:**

We can perform DFS from the cells adjacent to each ocean. Cells that can reach both oceans are the intersections of the cells visited from both DFS traversals.

```python
class Solution:
    def pacificAtlantic(self, heights):
        if not heights:
            return []
        m, n = len(heights), len(heights[0])
        pacific_visited = [[False for _ in range(n)] for _ in range(m)]
        atlantic_visited = [[False for _ in range(n)] for _ in range(m)]
        
        def dfs(x, y, visited):
            visited[x][y] = True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < m and 0 <= ny < n and
                    not visited[nx][ny] and
                    heights[nx][ny] >= heights[x][y]):
                    dfs(nx, ny, visited)
        
        # Perform DFS from Pacific Ocean borders
        for i in range(m):
            dfs(i, 0, pacific_visited)
            dfs(i, n - 1, atlantic_visited)
        for j in range(n):
            dfs(0, j, pacific_visited)
            dfs(m - 1, j, atlantic_visited)
        
        # Find cells that can reach both oceans
        result = []
        for i in range(m):
            for j in range(n):
                if pacific_visited[i][j] and atlantic_visited[i][j]:
                    result.append([i, j])
        return result
```

---

### Breadth-First Search (BFS)

#### Idea

Breadth-First Search (BFS) explores the neighbor nodes first before moving to the next level neighbors. It uses a queue to keep track of the nodes to be explored next. BFS is useful for:

- Finding the shortest path in unweighted graphs
- Level-order traversal
- Connectivity checks

#### Template

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        # Process the current node here if needed
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

#### Example Problem: [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

**Problem Description:**

Given a 2D grid where:

- `0` represents an empty cell.
- `1` represents a fresh orange.
- `2` represents a rotten orange.

Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.

**Solution:**

We can use BFS to model the spread of rot from rotten oranges to fresh ones.

```python
class Solution:
    def orangesRotting(self, grid):
        from collections import deque
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        
        # Initialize the queue with positions of rotten oranges
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        
        if fresh == 0:
            return 0  # No fresh oranges to rot
        
        minutes = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        
        while queue:
            minutes += 1
            for _ in range(len(queue)):
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < rows and 0 <= ny < cols and
                        grid[nx][ny] == 1):
                        grid[nx][ny] = 2
                        fresh -= 1
                        queue.append((nx, ny))
            if fresh == 0:
                return minutes
        
        return -1
```

---

## Topological Sort

#### Idea

Topological sorting is the linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge `uv`, vertex `u` comes before `v` in the ordering. It's used for:

- Scheduling tasks
- Resolving dependencies

#### Template Using DFS

```python
def topological_sort(graph):
    visited = set()
    stack = []
    
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(vertex)
    
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]  # Reverse the stack to get the order
```

#### Example Problem: [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

**Problem Description:**

There are `n` courses labeled from `0` to `n-1`. Some courses have prerequisites. Return the ordering of courses you should take to finish all courses. If it's impossible, return an empty array.

**Solution:**

We can model courses and prerequisites as a graph and perform topological sorting using DFS.

```python
class Solution:  
    def findOrder(self, numCourses: int, prerequisites: List[List[int]])
        graph = defaultdict(list)  
        top_sort = []  
  
        for course, preq in prerequisites:  
            graph[course].append(preq)  
  
        UNVISITED = 0  
        VISITING = 1  
        VISITED = 2  
        states = [UNVISITED] * numCourses  
  
        def dfs(node):  
            state = states[node]  
  
            if state == VISITED:  
                return True  
            elif state == VISITING:  
                return False  
  
            states[node] = VISITING  
  
            for neigh in graph[node]:  
                if not dfs(neigh):  
                    return False  
  
            states[node] = VISITED  
            top_sort.append(node)  
            return True  
  
        for course in range(numCourses):  
            if not dfs(course):  
                return []  
  
        return top_sort
```

---

## Shortest Path Algorithms

### Dijkstra's Algorithm

#### Idea

Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph with non-negative weights.

#### Template

```python
import heapq

def dijkstra(graph, start):
    heap = [(0, start)]
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    while heap:
        current_distance, current_vertex = heapq.heappop(heap)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances
```

#### Example Problem: [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)

**Problem Description:**

Given a list of travel times as directed edges `times`, where `times[i] = (u, v, w)`, send a signal from a node `K` to all nodes in the network. Return how long it takes for all nodes to receive the signal. If it's impossible, return `-1`.

**Solution:**

We can use Dijkstra's algorithm to find the shortest time to reach all nodes.

```python
class Solution:
    def networkDelayTime(self, times, n, k):
        import heapq
        graph = {i: [] for i in range(1, n + 1)}
        for u, v, w in times:
            graph[u].append((v, w))
        
        heap = [(0, k)]
        distances = {}
        
        while heap:
            time, node = heapq.heappop(heap)
            if node in distances:
                continue
            distances[node] = time
            for neighbor, wt in graph[node]:
                if neighbor not in distances:
                    heapq.heappush(heap, (time + wt, neighbor))
        
        if len(distances) == n:
            return max(distances.values())
        return -1
```

---

## Minimum Spanning Tree

### Prim's Algorithm

#### Idea

Prim's algorithm finds a Minimum Spanning Tree (MST) for a weighted undirected graph by starting from an arbitrary vertex and continuously adding the smallest edge that connects a vertex in the MST to a vertex outside it.

#### Template

```python
import heapq

def prim(graph, start):
    visited = set([start])
    edges = [(weight, start, to) for to, weight in graph[start]]
    heapq.heapify(edges)
    mst = []
    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))
            for to_next, weight in graph[to]:
                if to_next not in visited:
                    heapq.heappush(edges, (weight, to, to_next))
    return mst
```

#### Example Problem: [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

**Problem Description:**

Given an array `points` where `points[i] = [xi, yi]` represents a point on the X-Y plane, return the minimum cost to make all points connected. The cost between two points is the Manhattan distance.

**Solution:**

We can use Prim's algorithm to find the MST connecting all points.

```python
class Solution:
    def minCostConnectPoints(self, points):
        import heapq
        n = len(points)
        visited = set()
        min_heap = [(0, 0)]  # (cost, point_index)
        res = 0
        while len(visited) < n:
            cost, i = heapq.heappop(min_heap)
            if i in visited:
                continue
            visited.add(i)
            res += cost
            x1, y1 = points[i]
            for j in range(n):
                if j not in visited:
                    x2, y2 = points[j]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    heapq.heappush(min_heap, (dist, j))
        return res
```

---

## Cycle Detection

#### Idea

Detecting cycles in graphs is essential for avoiding infinite loops and deadlocks. In an undirected graph, cycles can be detected using DFS by keeping track of visited nodes and their parents.

#### Template for Undirected Graph

```python
def has_cycle(graph):
    visited = set()
    
    def dfs(v, parent):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False
```

#### Example Problem: [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

**Problem Description:**

There are `n` courses labeled from `0` to `n-1`. Some courses have prerequisites. Determine if it's possible to finish all courses. If there is a cycle, it's impossible.

**Solution:**

We can detect cycles in the course prerequisite graph using DFS.

```python
class Solution:  
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:  
        graph = defaultdict(list)  
  
        for course, preq in prerequisites:  
            graph[course].append(preq)  
  
        UNVISITED = 0  
        VISITING = 1  
        VISITED = 2  
        states = [UNVISITED] * numCourses  
  
        def dfs(node):  
            state = states[node]  
  
            if state == VISITED:  
                return True  
            elif state == VISITING:  
                return False  
  
            states[node] = VISITING  
  
            for neigh in graph[node]:  
                if not dfs(neigh):  
                    return False  
  
            states[node] = VISITED  
            return True  
  
        for course in range(numCourses):  
            if not dfs(course):  
                return False  
  
        return True
```
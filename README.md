## 1 两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = dict()
        for i, num in enumerate(nums):
            if target-num in res.keys():
                return [i, res[target-num]]
            res.update({num: i})
        return []
```

Python中哈希表用字典，可用enumerate()优化循环代码

## 2 两数相加

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = tail = ListNode()
        carry = 0
        while l1 or l2:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            s = v1 + v2 + carry
            tail.next = ListNode(s % 10)
            tail = tail.next
            carry = s // 10
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carry >= 1:
            tail.next = ListNode(carry)
        return head.next
```

使用头节点来辅助链表操作，注意循环条件和进位carry

## 3 无重复字符的最长子串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l, n = 0, len(s)
        hash_table = set()
        res = 0
        if n == 0:
            return 0
        for r in range(n):
            while s[r] in hash_table:
                hash_table.remove(s[l])
                l += 1
            res = max(res, r-l+1)
            hash_table.add(s[r])
        return res
```

滑动窗口，用左右指针和哈希表，注意更新哈希表和左指针

## 5 最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        start, max_len = 0, 1
        if n < 2:
            return s
        
        dp = [[False] * n for _ in range(n)]
        # length == 1, dp[i][i] = True
        for i in range(n):
            dp[i][i] = True
        # loop for length of s
        for L in range(2, n+1):
            for i in range(n):
                j = L + i - 1
                # bound
                if j > n-1:
                    break
                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    # length == 2, dp(i,i+1) = (s[i] == s[i+1])
                    if L == 2:
                        dp[i][j] = True
                    # length > 2, dp(i,j) = dp(i+1,j-1) \cap (s[i] == s[j])
                    else:
                        dp[i][j] = dp[i+1][j-1]
                if dp[i][j] == True and L > max_len:
                    start = i
                    max_len = L
        return s[start: start+max_len]
```

动态规划，注意初始情况和递推，循环条件是子串的长度，判断条件是字符是否相等

## 11 盛最多水的容器

给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        max_area = 0
        while left < right:
            if height[left] <= height[right]:
                area = (right-left)*height[left]
                left += 1
            else:
                area = (right-left)*height[right]
                right -= 1
            max_area = max(area, max_area)
        return max_area
```

双指针，盛水存在短板效应，注意细节面积计算公式

## 15 三数之和

给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []

        nums.sort()

        for first in range(n):
            # 需要和上一次枚举的数不相同
            if first > 0 and nums[first] == nums[first-1]:
                continue
            target = -nums[first]
            third = n - 1
            for second in range(first+1, n):
                # 需要和上一次枚举的数不相同
                if second > first+1 and nums[second] == nums[second-1]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    res.append([nums[first], nums[second], nums[third]])
        return res
```

排序+双指针，通过双指针优化第三重循环，通过判断来去掉重复解

## 17 电话号码的字母组合

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        mappings = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }
        n = len(digits)
        if n == 0:
            return []

        def backtrack(index):
            # 结束状态
            if index == n:
                conbinations.append(''.join(conbination))
            else:
                # 遍历所有可能的下一状态
                for letter in mappings[digits[index]]:
                    conbination.append(letter)
                    # 对新状态进行回溯
                    backtrack(index+1)
                    # 状态重置，返回原状态
                    conbination.pop()
        
        conbination = []
        conbinations = []
        backtrack(0)
        return conbinations


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        mappings = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }
        n = len(digits)
        if n == 0:
            return []

        def backtrack(index, seq):
            # 结束状态
            if index == n:
                conbinations.append(seq)
            else:
                for letter in mappings[digits[index]]:
                    backtrack(index+1, seq + letter)
        
        conbinations = []
        backtrack(0, '')
        return conbinations
```

> 当题目中出现 “所有组合” 等类似字眼时，我们第一感觉就要想到用回溯。
>
> 回溯算法强调了在状态空间特别大的时候，只用一份状态变量去搜索所有可能的状态，在搜索到符合条件的解的时候，通常会做一个拷贝，这就是为什么经常在递归终止条件的时候，有`res.add(new ArrayList<>(path));` 这样的代码。正是因为全程使用一份状态变量，因此它就有「恢复现场」和「撤销选择」的需要。

写了两种实现方法，前一种使用全局变量，在回溯时需要pop，第二种使用`+` 生成了新的字符串（Java 和 Python 里），每次往下面传递的时候，都是新字符串，因此在搜索的时候不用回溯。

## 19 删除链表的倒数第N个结点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # dummy node
        dummy = ListNode(0, head)
        
        length = 0
        while head:
            length += 1
            head = head.next

        current = dummy
        for _ in range(length-n):
            current = current.next
        current.next = current.next.next
        return dummy.next

    
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # dummy node
        dummy = ListNode(0, head)
        
        stack = []
        current = dummy
        while current:
            stack.append(current)
            current = current.next
        
        for _ in range(n):
            stack.pop()
        pre = stack[-1]
        pre.next = pre.next.next
        return dummy.next
```

两种做法：计算链表长度、栈，第一种做法比较直接，但要注意循环条件，第二种做法更好。对于链表的题，尤其是需要修改链表的，最好都先在head前面建一个dummy，因为可能存在对head操作的情况，使用dummy就不需要对head做特殊的判断

## 20 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        mappings = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        if len(s) == 1:
            return False
        
        for c in s:
            if c in mappings.keys():
                stack.append(c)
            else:
                if len(stack) != 0:
                    top = stack[-1]
                    stack.pop()
                    if mappings[top] != c:
                        return False
                else:
                    return False
        if len(stack) != 0:
            return False
        else:
            return True
```

栈的经典应用

## 21 合并两个有序链表

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy  = ListNode()
        current = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        if list1:
            while list1:
                current.next = list1
                list1 = list1.next
                current = current.next
        elif list2:
            while list2:
                current.next = list2
                list2 = list2.next
                current = current.next
        return dummy.next

    
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
```

两种做法：递归和迭代，用递归更好看，但要注意Python中调用类方法的方式self.func()

## 22 括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(left, right, n, seq):
            # 结束条件
            if left == n and right == n:
                res.append(seq)
                return
            if left < n:
                dfs(left + 1, right, n, seq + '(')
            if right < n and right < left:
                dfs(left, right + 1, n, seq + ')')
        
        res = []
        dfs(0, 0, n, '')
        return res

    
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(left, right, n):
            # 结束条件
            if left == n and right == n:
                res.append(''.join(seq))
                return
            if left < n:
                seq.append('(')
                dfs(left + 1, right, n)
                seq.pop()
            if right < n and right < left:
                seq.append(')')
                dfs(left, right + 1, n)
                seq.pop()
        
        seq = []
        res = []
        dfs(0, 0, n)
        return res
```

回溯法的基本结构：DFS+状态重置+剪枝，注意两种写法。

括号序列合法的充要条件：任意前缀左括号数量大于等于右括号数量；左右括号数量相等

- 添加左括号条件：只要左括号总数不超过n
- 添加右括号条件：受到前缀中左括号数量约束，需满足右括号总数不超过n并且小于左括号数量
- 结束条件：左右括号数量都等于n

## 33 搜索旋转排序数组

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            # [l, mid]有序
            elif nums[mid] >= nums[l]:
                # 是否在有序区间
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # [mid, r]有序
            else:
                # 是否在有序区间
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

O(log n)算法考虑二分。因为我们将旋转数组从中间分开成左右两部分的时候，一定有一部分的数组是有序的，所以需要根据mid来对数组分段讨论，确定边界更新情况。

> 搞懂这个题的精髓在于三个定理
>
> 定理一：只有在顺序区间内才可以通过区间两端的数值判断target是否在其中。
>
> 定理二：判断顺序区间还是乱序区间，只需要对比 left 和 right 是否是顺序对即可，left <= right，顺序区间，否则乱序区间。
>
> 定理三：每次二分都会至少存在一个顺序区间。
>
> 通过不断的用Mid二分，根据定理二，将整个数组划分成顺序区间和乱序区间，然后利用定理一判断target是否在顺序区间，如果在顺序区间，下次循环就直接取顺序区间，如果不在，那么下次循环就取乱序区间。

## 34 在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def searchFirst(nums, target):
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] == target:
                    # 不能直接返回
                    # 判断左边一个元素是否等于mid元素或mid元素是否已经是数组最左元素
                    if mid == 0 or nums[mid - 1] != nums[mid]:
                        return mid
                    # 第一个位置还在mid左边
                    else:
                        r = mid - 1
                elif nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
            return -1
        
        def searchLast(nums, target):
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] == target:
                    if mid == len(nums) - 1 or nums[mid] != nums[mid + 1]:
                        return mid
                    else:
                        l = mid + 1
                elif nums[mid] > target:
                    r = mid - 1
                else:
                     l = mid + 1
            return -1
        
        first = searchFirst(nums, target)
        last = searchLast(nums, target)
        return [first, last]
```

分别二分找元素的第一个位置和最后一个位置。和传统二分相比，当mid元素等于target后不能直接返回，而需要先判断前（后）一个元素是否等于target，保证取到的是第一个（最后一个）位置，否则还得继续更新边界来找值。

## 39 组合总和

给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(candidates, target, combination, begin):
            if target < 0:
                return
            if target == 0:
                combinations.append(combination)
                return
            for i in range(begin, len(candidates)):
                # 剪枝
                if target - candidates[i] < 0:
                    break
                backtrack(candidates, target-candidates[i], combination+[candidates[i]], i)
        
        # 顺序遍历，需要先排序
        candidates.sort()
        combinations = []
        combination = []
        backtrack(candidates, target, combination, 0)
        return combinations
```

回溯

> 根据示例 1：输入: `candidates = [2, 3, 6, 7]`，`target = 7`
>
> 候选数组里有 2，如果找到了组合总和为 7 - 2 = 5 的所有组合，再在之前加上 2 ，就是 7 的所有组合；同理考虑 3，如果找到了组合总和为 7 - 3 = 4 的所有组合，再在之前加上 3 ，就是 7 的所有组合，依次这样找下去。
>
> 遇到这一类相同元素不计算顺序的问题，我们在搜索的时候就需要 按某种顺序搜索。具体的做法是：每一次搜索的时候设置 下一轮搜索的起点 begin。
>
> 
>
> 什么时候使用 used 数组，什么时候使用 begin 变量：排列问题，讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为不同列表时），需要记录哪些数字已经使用过，此时用 used 数组；组合问题，不讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为相同列表时），需要按照某种顺序搜索，此时使用 begin 变量。
>

## 40 组合总和 II

给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

注意：解集不能包含重复的组合。 

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(candidates, target, combination, begin):
            if target == 0:
                res.append(combination[:])
                return
            for i in range(begin, len(candidates)):
                #大剪枝：target 减去 candidates[i] 小于 0，减去后面的 candidates[i + 1]、candidates[i + 2] 肯定也小于 0，因此用 break
                if target < candidates[i]:
                    break
                # 小剪枝：同一层相同数值的结点，从第 2 个开始，候选数更少，结果一定发生重复，因此跳过，用 continue
                if i > begin and candidates[i] == candidates[i-1]:
                    continue
                combination.append(candidates[i])
                backtrack(candidates, target-candidates[i], combination, i+1)
                combination.pop()

        candidates.sort()
        res = []
        combination = []
        backtrack(candidates, target, combination, 0)
        return res
```

回溯

> 区别：第 39 题candidates 中的数字可以无限制重复被选取；第 40 题candidates 中的每个数字在每个组合中只能使用一次。
>
> 相同点：相同数字列表的不同排列视为一个结果。
>
> 如何去掉重复的集合：类似于39题，不重复就需要按 **顺序** 搜索， **在搜索的过程中检测分支是否会出现重复结果** 。注意：这里的顺序不仅仅指数组 `candidates` 有序，还指按照一定顺序搜索结果。
>
> 由第 39 题我们知道，数组 candidates 有序，也是 深度优先遍历 过程中实现「剪枝」的前提。
> 将数组先排序的思路来自于这个问题：去掉一个数组中重复的元素。很容易想到的方案是：先对数组 升序 排序，重复的元素一定不是排好序以后相同的连续数组区域的第 1 个元素。也就是说，剪枝发生在：同一层数值相同的结点第 2、3 ... 个结点，因为数值相同的第 1 个结点已经搜索出了包含了这个数值的全部结果，同一层的其它结点，候选数的个数更少，搜索出的结果一定不会比第 1 个结点更多，并且是第 1 个结点的子集。

## 46 全排列

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, path, used):
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if not used[i]:
                    path.append(nums[i])
                    used[i] = True
                    backtrack(nums, path, used)
                    path.pop()
                    used[i] = False

        used = [False] * len(nums)
        path, res = [], []
        backtrack(nums, path, used)
        return res
```

回溯。使用一个used数组来辅助，注意循环是从0到len(nums)-1

> 变量 path 所指向的列表 在深度优先遍历的过程中只有一份 ，深度优先遍历完成以后，回到了根结点，成为空列表。
>
> 在 Java 中，参数传递是 值传递，对象类型变量在传参的过程中，复制的是变量的地址。这些地址被添加到 res 变量，但实际上指向的是同一块内存地址，因此我们会看到 6 个空的列表对象。解决的方法很简单，在 res.add(path); 这里做一次拷贝即可。
>
> 
>
> 每一次尝试都「复制」，则不需要回溯
>
> 如果在每一个 非叶子结点 分支的尝试，都创建 新的变量 表示状态，那么
>
> - 在回到上一层结点的时候不需要「回溯」；
> - 在递归终止的时候也不需要做拷贝。
>   这样的做法虽然可以得到解，但也会创建很多中间变量，这些中间变量很多时候是我们不需要的，会有一定空间和时间上的消耗。
>
> 在一些字符串的搜索问题中（比如17题、22题），有时不需要回溯的原因是这样的：字符串变量在拼接的过程中会产生新的对象（针对 Java 和 Python 语言，其它语言我并不清楚）。如果您使用 Python 语言，会知道有这样一种语法：[1, 2, 3] + [4] 也是创建了一个新的列表对象。

## 48 旋转图像

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
```

找规律，推公式。对于矩阵中第 i 行的第 j 个元素，在旋转后，它出现在倒数第 i 列的第 j 个位置。

## 49 字母异位词分组

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_table = {} # sorted str: [str]
        for s in strs:
            sorted_s = ''.join(sorted(s))
            if sorted_s in hash_table.keys():
                hash_table[sorted_s].append(s)
            else:
                hash_table.update({sorted_s: [s]})
        return list(hash_table.values())
```

由于互为字母异位词的两个字符串包含的字母相同，因此对两个字符串分别进行排序之后得到的字符串一定是相同的，故可以将排序之后的字符串作为哈希表的键。

## 53 最大子数组和

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i]：以nums[i]结尾的连续子数组的最大和
        dp = [0] * n
        # dp[0] = nums[0]
        dp[0] = nums[0]

        for i in range(1, n):
            # dp[i] = dp[i-1] + nums[i], dp[i-1] > 0
            # dp[i] = nums[i], dp[i-1] <= 0
            dp[i] = max(dp[i-1]+nums[i], nums[i])
        return max(dp)
```

动态规划。`dp[i]`：表示以 `nums[i]` **结尾** 的 **连续** 子数组的最大和。根据状态的定义，由于 nums[i] 一定会被选取，并且以 nums[i] 结尾的连续子数组与以 nums[i - 1] 结尾的连续子数组只相差一个元素 nums[i] 。

假设数组 nums 的值全都严格大于 0，那么一定有 dp[i] = dp[i - 1] + nums[i]。

可是 dp[i - 1] 有可能是负数，于是分类讨论：如果 dp[i - 1] > 0，那么可以把 nums[i] 直接接在 dp[i - 1] 表示的那个数组的后面，得到和更大的连续子数组；如果 dp[i - 1] <= 0，那么 nums[i] 加上前面的数 dp[i - 1] 以后值不会变大。于是 dp[i] 「另起炉灶」，此时单独的一个 nums[i] 的值，就是 dp[i]。

## 55 跳跃游戏

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] < i:
                return False
            dp[i] = max(dp[i-1], i + nums[i])
        return True
    
# 滚动数组，优化空间复杂度
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        max_index = nums[0]
        for i in range(1, n):
            if max_index < i:
                return False
            max_index = max(max_index, i + nums[i])
        return True
```

动态规划

> dp[i] 表示从下标范围 [0,i] 中的任意下标出发可以到达的最大下标。
>
> 对于 1≤i<n，如果可以从下标 0 到达下标 i，则可以从下标 i 到达不超过下标 i+nums[i] 的任意位置，因此可以从下标 0 到达不超过下标 i+nums[i] 的任意位置。为了判断是否可以到达下标 n−1，需要分别计算从每个下标出发可以到达的最大下标。对于每个下标，需要首先判断是否可以从更小的下标到达该下标，然后计算从该下标出发可以到达的最大下标，因此可以使用动态规划计算从每个下标出发可以到达的最大下标。
>

## 56 合并区间

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 区间按左端点升序排列
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
```

> 如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。
>
> 我们用数组 merged 存储最终的答案。首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间加入 merged 数组中，并按顺序依次考虑之后的每个区间：如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组 merged 的末尾；否则，它们重合，我们需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，将其置为二者的较大值。
>

## 62 不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*n for _ in range(m)]
        # 初始化
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

动态规划。每个位置的路径 = 该位置左边的路径 + 该位置上边的路径

## 64 最小路径和

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0]*n for _ in range(m)]
        # 初始化
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[m-1][n-1]
```

动态规划

## 70 爬楼梯

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1

        dp = [0] * (n+1)
        # 初始化
        dp[1], dp[2] = 1, 2
        
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

class Solution:
    def climbStairs(self, n: int) -> int:
        # 初始化，认为到达第0阶只有一种方法
        dp = [1] * (n+1)  
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

动态规划。爬到第 x 级台阶的方案数是爬到第 x−1 级台阶的方案数和爬到第 x−2 级台阶的方案数的和。很好理解，因为每次只能爬 1 级或 2 级。

## 75 颜色分类

给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

```python
# 做法一：统计出数组中 0,1,2 的个数，再根据它们的数量，重写整个数组
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count_dict = {}
        for i in range(3):
            count_dict[i] = nums.count(i)
        index = 0
        for i in range(3):
            for _ in range(count_dict[i]):
                nums[index] = i
                index += 1
                
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ptr = 0
        n = len(nums)
        for curr in range(n):
            if nums[curr] == 0:
                nums[ptr], nums[curr] = nums[curr], nums[ptr]
                ptr += 1
        for curr in range(ptr, n):
            if nums[curr] == 1:
                nums[ptr], nums[curr] = nums[curr], nums[ptr]
                ptr += 1
                
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        p0, p2 = 0, len(nums)-1
        i = 0
        while i <= p2:
            while i <= p2 and nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i += 1
```

> 做法二：单指针（两次遍历）
>
> 在第一次遍历中，我们将数组中所有的 0 交换到数组的头部。在第二次遍历中，我们将数组中所有的 1 交换到头部的 0 之后。此时，所有的 2 都出现在数组的尾部，这样我们就完成了排序。具体地，我们使用一个指针 ptr 表示「头部」的范围，ptr 中存储了一个整数，表示数组 nums 从位置 0 到位置 ptr−1 都属于「头部」。ptr 的初始值为 0，表示还没有数处于「头部」。
>
> 在第一次遍历中，我们从左向右遍历整个数组，在第二次遍历中，我们从「头部」开始，从左向右遍历整个数组。
>
> 做法三：双指针
>
> 使用指针 p0 来交换 0，p2 来交换 2。此时，p0 的初始值仍然为 0，而 p2 的初始值为 n−1。在遍历的过程中，我们需要找出所有的 0 交换至数组的头部，并且找出所有的 2 交换至数组的尾部，头部为[0, p0-1]，尾部为[p2+1, n-1]。由于此时其中一个指针 p2 是从右向左移动的，因此当我们在从左向右遍历整个数组时，如果遍历到的位置超过了 p2 ，那么就可以直接停止遍历了（因为后面的尾部已经有序）
>
> 当我们将 nums[i] 与 nums[p2] 进行交换之后，新的 nums[i] 可能仍然是 2，也可能是 0。然而此时我们已经结束了交换，开始遍历下一个元素 nums[i+1]，不会再考虑 nums[i] 了，这样我们就会得到错误的答案。因此，当我们找到 2 时，我们需要不断地将其与 nums[p2] 进行交换，直到新的 nums[i] 不为 2。此时，如果 nums[i] 为 0，那么对应着第一种情况；如果 nums[i] 为 1，那么就不需要进行任何后续的操作。（原因在于遍历是从左到右，我们并不知道从后面p2交换过来的元素是什么元素）

## 78 子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(index):
            # index位置到达数组末尾
            if index == n:
                res.append(path[:])
                return
            # 不选index元素
            backtrack(index+1)
            # 选index元素
            path.append(nums[index])
            backtrack(index+1)
            path.pop()
        
        n = len(nums)
        path = []
        res = []
        backtrack(0)
        return res
```

子集型回溯，每个元素都可以选或不选，都需要分别递归。

## 79 单词搜索

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def check(board, i, j,used):
            # 递归时元素的坐标是否超过边界和被使用过
            if 0<= i < len(board) and 0 <= j < len(board[0]):
                if used[i][j] == 0:
                    return True
                else:
                    return False
            else:
                return False

        def dfs(board, i, j, word, used):
            if len(word) == 0:
                return True
            for e in [(i-1,j), (i+1,j),(i,j-1),(i,j+1)]:
                if not check(board, e[0], e[1], used):
                    continue
                else:
                    if board[e[0]][e[1]] == word[0]:
                        used[e[0]][e[1]] = 1
                        if dfs(board, e[0], e[1], word[1:], used):
                            return True
                        else:
                            # 回溯
                            used[e[0]][e[1]] = 0
                 
        m, n = len(board), len(board[0])
        # 同一元素不允许重复使用
        used = [[0] * n for _ in range(m)]
        # 遍历矩阵
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    used[i][j] = 1
                    if dfs(board, i, j, word[1:], used):
                        return True
                    else:
                        # 回溯
                        used[i][j] = 0
        return False
```

dfs+回溯。首次拿下回溯中等题

## 88 合并两个有序数组

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

**注意：**最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0` ，应忽略。`nums2` 的长度为 `n` 。

```python
# 做法一：直接合并后排序
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[m: ] = nums2
        nums1.sort()
        
# 做法二：双指针，用临时数组存中间结果
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        temp = []
        p1 = p2 = 0
        while p1 < m or p2 < n:
            if p1 == m:
                temp.append(nums2[p2])
                p2 += 1
            elif p2 == n:
                temp.append(nums1[p1])
                p1 += 1
            elif nums1[p1] <= nums2[p2]:
                temp.append(nums1[p1])
                p1 += 1
            else:
                temp.append(nums2[p2])
                p2 += 1
        nums1[:] = temp
        
# 做法三：逆向双指针
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m-1, n-1
        tail = m+n-1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1:
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] < nums2[p2]:
                nums1[tail] = nums2[p2]
                p2 -= 1
            else:
                nums1[tail] = nums1[p1]
                p1 -= 1
            tail -= 1
```

> 方法二中，之所以要使用临时变量，是因为如果直接合并到数组 nums1 中，nums1 中的元素可能会在取出之前被覆盖。那么如何直接避免覆盖 nums1中的元素呢？观察可知，nums1的后半部分是空的，可以直接覆盖而不会影响结果。因此可以指针设置为从后向前遍历，每次取两者之中的较大者放进 nums1的最后面。
>

## 94 二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        
        res = []
        dfs(root)
        return res
```

递归，dfs

## 96 不同的二叉搜索树

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        # 初始化
        dp[0], dp[1] = 1, 1
        # 遍历
        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1] * dp[i-j]
        return dp[n]
```

动态规划

> 二叉搜索树(Binary Search Tree，BST)，也称为二叉排序树或二叉查找树。
>
> 相较于普通的二叉树，非空的二叉搜索树有如下性质：
>
> 1. 非空**左子树**的所有**键值小于其根结点**的键值；
> 2. 非空**右子树**的所有**键值大于其根结点**的键值；
> 3. **左右子树均为二叉搜索树**；
> 4. **树中没有键值相等的结点**。
>
> 给定一个有序序列 1⋯n，为了构建出一棵二叉搜索树，我们可以遍历每个数字 i，将该数字作为树根，将 1⋯(i−1) 序列作为左子树，将 (i+1)⋯n 序列作为右子树。接着我们可以按照同样的方式递归构建左子树和右子树。在上述构建的过程中，由于根的值不同，因此我们能保证每棵二叉搜索树是唯一的。
>
> 对于边界情况，当序列长度为 1（只有根）或为 0（空树）时，只有一种情况。给定序列 1⋯n，我们选择数字 i 作为根，则根为 i 的所有二叉搜索树的集合是左子树集合和右子树集合的笛卡尔积，对于笛卡尔积中的每个元素，加上根节点之后形成完整的二叉搜索树。
>

## 98 验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 错误代码，只判断了根节点和左右子节点的范围，应该判断整个左右子树的范围，所以需要传递一个上下界
# 出错输入：[5,4,6,null,null,3,7]
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if root.left:
            if root.left.val < root.val:
                self.isValidBST(root.left)
            else:
                return False
        if root.right:
            if root.val < root.right.val:
                self.isValidBST(root.right)
            else:
                return False
        return True
    
# 做法一：递归
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def check(node, lower, upper):
            if not node:
                return True
 
            val = node.val
            if not lower < val < upper:
                return False
            if not check(node.left, lower, val):
                return False
            if not check(node.right, val, upper):
                return False
            return True
        
        return check(root, float(-inf), float(inf))
    
# 做法二：中序遍历    
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            inorder.append(root.val)
            dfs(root.right)

        inorder = []
        dfs(root)
        for i in range(1, len(inorder)):
            if inorder[i-1] >= inorder[i]:
                return False
        return True
```

> 做法一：递归
>
> 这启示我们设计一个递归函数 helper(root, lower, upper) 来递归判断，函数表示考虑以 root 为根的子树，判断子树中所有节点的值是否都在 (l,r) 的范围内（注意是开区间）。如果 root 节点的值 val 不在 (l,r) 的范围内说明不满足条件直接返回，否则我们要继续递归调用检查它的左右子树是否满足，如果都满足才说明这是一棵二叉搜索树。
>
> 那么根据二叉搜索树的性质，在递归调用左子树时，我们需要把上界 upper 改为 root.val，即调用 helper(root.left, lower, root.val)，因为左子树里所有节点的值均小于它的根节点的值。同理递归调用右子树时，我们需要把下界 lower 改为 root.val，即调用 helper(root.right, root.val, upper)。
>
> 函数递归调用的入口为 helper(root, -inf, +inf)， inf 表示一个无穷大的值
>
> 
>
> 做法二：中序遍历
>
> 二叉搜索树「中序遍历」得到的值构成的序列一定是升序的，这启示我们在中序遍历的时候实时检查当前节点的值是否大于前一个中序遍历到的节点的值即可。如果均大于说明这个序列是升序的，整棵树是二叉搜索树，否则不是

## 101 对称二叉树

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def check(left, right):
            # 两个节点都为空，对称
            if (not left) and (not right):
                return True
            # 其中一个为空，另一个不为空，不对称
            if (not left) or (not right):
                return False
            # 两边子树的根节点值不相等，不对称
            if left.val != right.val:
                return False
            return check(left.left, right.right) and check(left.right, right.left)
        
        return check(root.left, root.right)
```

> 怎么判断一棵树是不是对称二叉树？ 答案：如果所给根节点，为空，那么是对称。如果不为空的话，当他的左子树与右子树对称时，他对称
>
> 那么怎么知道左子树与右子树对不对称呢？在这我直接叫为左树和右树 答案：如果左树的左孩子与右树的右孩子对称，左树的右孩子与右树的左孩子对称，那么这个左树和右树就对称

## 102 二叉树的层序遍历

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        queue = [root]
        res = []
        while queue:
            # 获取当前队列的长度，这个长度相当于 当前这一层的节点个数
            size = len(queue)
            level = []
            # 一次处理一整层的节点
            for _ in range(size):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res

# 附：广度优先搜索
def bfs(root):
    if not root:
        return []
    
    queue = [root]
    res = []
    while queue:
        node = queue.pop(0)
        res.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return res
```

> 层序遍历就是把二叉树分层，然后每一层从左到右遍历。乍一看来，这个遍历顺序和 BFS 是一样的，我们可以直接用 BFS 得出层序遍历结果。然而，层序遍历要求的输入结果和 BFS 是不同的。层序遍历要求我们区分每一层，也就是返回一个二维数组。而 BFS 的遍历结果是一个一维数组，无法区分每一层。
>
> 需要稍微修改一下代码，在每一层遍历开始前，先记录队列中的结点数量 *n*（也就是这一层的结点数量），然后一口气处理完这一层的 *n* 个结点（观察这个算法，可以归纳出一个循环不变式：第 *i* 次迭代前，队列中的所有元素就是第 *i* 层的所有元素，并且按照从左向右的顺序排列）

## 104 二叉树的最大深度

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 做法一：BFS
# 这道题只需要层序遍历时，每层把res加一
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        res = 0
        queue = [root]
        while queue:
            size = len(queue)
            res += 1
            for _ in range(size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res
    
# 做法二：DFS
# 如果我们知道了左子树和右子树的最大深度 l 和 r，那么该二叉树的最大深度即为 max⁡(l,r)+1，而左子树和右子树的最大深度又可以以同样的方式进行计算
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1
```

## 105 从前序与中序遍历序列构造二叉树

给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def build(preorder, inorder):
            # preorder第一个元素是该子树的根节点值
            root_val = preorder.pop(0)
            root = TreeNode(root_val)
            
            # 找到在inorder中的位置，前面为左子树节点值，后面为右子树节点值
            inorder_root_index = inorder.index(root_val)
            inorder_left = inorder[: inorder_root_index]
            inorder_right = inorder[inorder_root_index+1: ]
            
            if inorder_left:
                root.left = build(preorder, inorder_left)
            else:
                root.left = None
            if inorder_right:
                root.right = build(preorder, inorder_right)
            else:
                root.right = None
            return root
        
        if not preorder or not inorder:
            return None
        return build(preorder, inorder)
```

> 只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。
>
> 这样以来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，我们就可以递归地对构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。
>
> 在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。我们可以考虑使用哈希表来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 O(1) 的时间对根节点进行定位了。（Python里直接用list.index()函数了，不然应该先hash_table = {element: index for index, element in enumerate(inorder)}）
>

## 114 二叉树展开为链表

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/先序遍历/6442839?fr=aladdin) 顺序相同。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 做法一：直接先序遍历再重建树，不是原地算法
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """  
        def dfs(root):
            if not root:
                return
            preorder.append(root.val)
            dfs(root.left)
            dfs(root.right)
        
        def build(preorder):
            if not preorder:
                return None
            val = preorder.pop(0)
            curr = TreeNode(val)
            curr.right = build(preorder)
            return curr
        
        if not root:
            return
        preorder = []
        dfs(root)
        root.left = None
        root.right = build(preorder[1:])
        
# 做法一优化：先序遍历直接保存整个节点，再迭代修改节点指针，现在是原地算法，但空间复杂度是O(n)
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def dfs(root):
            if not root:
                return
            preorder.append(root)
            dfs(root.left)
            dfs(root.right)
        
        preorder = []
        dfs(root)
        for i in range(1, len(preorder)):
            prev, curr = preorder[i-1], preorder[i]
            prev.left = None
            prev.right = curr
        
# 做法二：前序遍历和展开同步进行，原地算法，但空间复杂度O(n)
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        
        prev = None
        stack = [root]
        while stack:
            curr = stack.pop()
            if prev:
                prev.left = None
                prev.right = curr
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)
            prev = curr
            
# 做法三：一个节点的左子树的最右的节点是右子树根节点的前驱节点，原地算法，空间复杂度O(1)
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root:
            # 左子树不存在
            if not root.left:
                root = root.right
            else:
                # 找左子树最右节点
                prev = root.left
                while prev.right:
                    prev = prev.right
                # 右子树接到左子树最右节点后面
                prev.right = root.right
                # 左子树接到根节点右边
                root.right = root.left
                # 关键细节
                root.left = None
                # 继续往下
                root = root.right
```

> 做法二：前序遍历和展开同步进行
>
> 使用方法一的前序遍历，由于将节点展开之后会破坏二叉树的结构而丢失子节点的信息，因此前序遍历和展开为单链表分成了两步。能不能在不丢失子节点的信息的情况下，将前序遍历和展开为单链表同时进行？
>
> 之所以会在破坏二叉树的结构之后丢失子节点的信息，是因为在对左子树进行遍历时，没有存储右子节点的信息，在遍历完左子树之后才获得右子节点的信息。只要对前序遍历进行修改，在遍历左子树之前就获得左右子节点的信息，并存入栈内，子节点的信息就不会丢失，就可以将前序遍历和展开为单链表同时进行。
>
> 该做法不适用于递归实现的前序遍历，只适用于迭代实现的前序遍历。修改后的前序遍历的具体做法是，每次从栈内弹出一个节点作为当前访问的节点，获得该节点的子节点，如果子节点不为空，则依次将右子节点和左子节点压入栈内（注意入栈顺序）。
>
> 展开为单链表的做法是，维护上一个访问的节点 prev，每次访问一个节点时，令当前访问的节点为 curr，将 prev 的左子节点设为 null 以及将 prev 的右子节点设为 curr，然后将 curr 赋值给 prev，进入下一个节点的访问，直到遍历结束。需要注意的是，初始时 prev 为 null，只有在 prev 不为 null 时才能对 prev 的左右子节点进行更新。
>
> 做法三：一个节点的左子树的最右的节点是右子树根节点的前驱节点
>
> 将左子树插入到右子树的地方；将原来的右子树接到左子树的最右边节点；考虑新的右子树的根节点，一直重复上边的过程，直到新的右子树为 null
>

## 121 买卖股票的最佳时机

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i]：第i天卖出的利润
        dp = [0] * n
        # 前i天的最低价格
        min_price = prices[0]
        for i in range(1, n):
            dp[i] = max(prices[i]-min_price, 0)
            min_price = min(min_price, prices[i])
        return max(dp)
```

动态规划

## 128 最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        # 哈希表去重，优化查找
        nums = set(nums)

        for num in nums:
            # 只有前驱数不在时才开始找，否则一定不是最长序列
            if num-1 not in nums:
                curr_num = num
                curr_length = 1
                # 遍历找后继数
                while curr_num + 1 in nums:
                    curr_num += 1
                    curr_length += 1
                res = max(curr_length, res)
        return res
```

> 我们考虑枚举数组中的每个数 x，考虑以其为起点，不断尝试匹配 x+1,x+2,⋯ 是否存在，假设最长匹配到了 x+y，那么以 x 为起点的最长连续序列即为 x,x+1,x+2,⋯ ,x+y，其长度为 y+1，我们不断枚举并更新答案即可
>
> 对于匹配的过程，暴力的方法是 O(n) 遍历数组去看是否存在这个数，但其实更高效的方法是用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化至 O(1) 的时间复杂度
>
> 仅仅是这样我们的算法时间复杂度最坏情况下还是会达到 O(n^2)（即外层需要枚举 O(n) 个数，内层需要暴力匹配 O(n) 次），无法满足题目的要求。但仔细分析这个过程，我们会发现其中执行了很多不必要的枚举，如果已知有一个 x,x+1,x+2,⋯ ,x+y 的连续序列，而我们却重新从 x+1，x+2 或者是 x+y 处开始尝试匹配，那么得到的结果肯定不会优于枚举 x 为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。
>
> 那么怎么判断是否跳过呢？由于我们要枚举的数 x 一定是在数组中不存在前驱数 x−1 的，不然按照上面的分析我们会从 x−1 开始尝试匹配，因此我们每次在哈希表中检查是否存在 x−1 即能判断是否需要跳过了。
>
> 增加了判断跳过的逻辑之后，时间复杂度是多少呢？外层循环需要 O(n) 的时间复杂度，只有当一个数是连续序列的第一个数的情况下才会进入内层循环，然后在内层循环中匹配连续序列中的数，因此数组中的每个数只会进入内层循环一次。根据上述分析可知，总时间复杂度为 O(n)，符合题目要求。
>

## 130 被围绕的区域

给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def check(board, i, j):
            if 0 <= i < len(board) and 0 <= j < len(board[0]):
                return True
            else:
                return False
        
        def dfs(board, i, j):
            board[i][j] = 'A'
                
            for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if check(board, k[0], k[1]) and board[k[0]][k[1]] == 'O':
                    dfs(board, k[0], k[1])
                
        m = len(board)
        n = len(board[0])
        for i in range(m):
            if board[i][0] == 'O':
                dfs(board, i, 0)
            if board[i][n-1] == 'O':
                dfs(board, i, n-1)
        for j in range(n):
            if board[0][j] == 'O':
                dfs(board, 0, j)
            if board[m-1][j] == 'O':
                dfs(board, m-1, j)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'A':
                    board[i][j] = 'O'
                elif board[i][j] == 'O':
                    board[i][j] = 'X'
```

与岛屿题思路不一样，不要硬套模板

> 本题给定的矩阵中有三种元素：字母 X；被字母 X 包围的字母 O；没有被字母 X 包围的字母 O。
>
> 本题要求将所有被字母 X 包围的字母 O都变为字母 X ，但很难判断哪些 O 是被包围的，哪些 O 不是被包围的。
>
> 注意到题目解释中提到：任何边界上的 O 都不会被填充为 X。 我们可以想到，所有的不被包围的 O 都直接或间接与边界上的 O 相连。我们可以利用这个性质判断 O 是否在边界上，具体地说：对于每一个边界上的 O，我们以它为起点，标记所有与它直接或间接相连的字母 O；最后我们遍历这个矩阵，对于每一个字母：如果该字母被标记过，则该字母为没有被字母 X 包围的字母 O，我们将其还原为字母 O；如果该字母没有被标记过，则该字母为被字母 X 包围的字母 O，我们将其修改为字母 X。

## 136 只出现一次的数字

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

```python
# 哈希表
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 统计元素次数，再遍历输出，时间复杂度O(n)，空间复杂度O(n)
        hash_table = {} # value: count
        for num in nums:
            if num in hash_table.keys():
                hash_table[num] += 1
            else:
                hash_table[num] = 1
        for k in hash_table.keys():
            if hash_table[k] == 1:
                return k
# 排序            
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # 先排序，再遍历找到只有一个元素的位置，时间复杂度O(nlogn)，空间复杂度O(1)
        nums.sort()
        index = 0
        while index < len(nums) - 1:
            if nums[index] == nums[index+1]:
                index += 2
            else:
                break
        return nums[index]
    
# 位运算
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1, len(nums)):
            res ^= nums[i]
        return res
```

> 使用位运算。对于这道题，可使用异或运算 $\oplus$。异或运算有以下三个性质:
>
> 任何数和 0 做异或运算，结果仍然是原来的数，即 $a \oplus 0=a$；任何数和其自身做异或运算，结果是 0，即 $a \oplus a=0$；异或运算满足交换律和结合律，即$a \oplus b \oplus a=b \oplus a \oplus a=b \oplus (a \oplus a)=b \oplus0=b$
>
> 数组中的全部元素的异或运算结果即为数组中只出现一次的数字

## 139 单词拆分

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。请你判断是否可以利用字典中出现的单词拼接出 `s` 。

**注意：**不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True

        for i in range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    # 剪枝，能找到一种表示组合即可
                    break
        return dp[n]
```

动态规划，注意循环边界

> dp[i] 表示字符串 s 前 i 个字符组成的字符串 s[0..i−1] 是否能被空格拆分成若干个字典中出现的单词
>
> *dp*[*i*]=*dp*[*j*] && *check*(*s*[*j*..*i*−1])，其中 check(s[j..i−1]) 表示子串 s[j..i−1] 是否出现在字典中
>
> *dp*[0]=true 表示空串且合法

## 141 环形链表

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

*如果链表中存在环* ，则返回 `true` 。 否则，返回 `false` 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 做法一：哈希表，使用哈希表来存储所有已经访问过的节点
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        hash_table = set()
        while head:
            if head in hash_table:
                return True
            hash_table.add(head)
            head = head.next
        return False

# 做法二：快慢指针
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = slow = head
        # 如果无环，则一定是fast先到链表尾
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            # 移动后再判断，因为初始化fast = slow = head
            if fast == slow:
                return True
        return False
```

> 我们定义两个指针，一快一慢。慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 head，而快指针在位置 head.next。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表。否则快指针将到达链表尾部，该链表不为环形链表
>

## 142 环形链表 II

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 做法一：哈希表
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        hash_table = set()
        while head:
            if head in hash_table:
                return head
            hash_table.add(head)
            head = head.next
        return None

# 做法二：快慢指针
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        # 判环
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            # fast和slow相遇
            # f = 2s, f = s + nb => s = nb
            if fast == slow:
                # slow再走a步
                while head != slow:
                    head = head.next
                    slow = slow.next
                return slow
        return None
```

> 当`fast == slow`时， 两指针在环中第一次相遇。下面分析此时 `fast` 与 `slow` 走过的步数关系
>
> 设链表共有 a+b 个节点，其中 链表头部到链表入口 有 a 个节点（不计链表入口节点）， 链表环 有 b 个节点（这里需要注意，a 和 b 是未知数）；设两指针分别走了 f，s 步，则有：
>
> 1. `fast` 走的步数是 `slow` 步数的 2 倍，即 f=2s；（**解析：** `fast` 每轮走 2 步）
> 2. fast 比 slow 多走了 n 个环的长度，即 f=s+nb；（ 解析： 双指针都走过 a 步，然后在环内绕圈直到重合，重合时 fast 比 slow 多走 环的长度整数倍 ）。
>
> 将以上两式相减得到 f=2nb，s=nb，即 fast 和 slow 指针分别走了 2n，n 个环的周长
>
> 如果让指针从链表头部一直向前走并统计步数k，那么所有 走到链表入口节点时的步数 是：k=a+nb ，即先走 a 步到入口节点，之后每绕 1 圈环（ b 步）都会再次到入口节点。而目前 slow 指针走了 nb 步。因此，我们只要想办法让 slow 再走 a 步停下来，就可以到环的入口。
>
> 但是我们不知道 a 的值，该怎么办？依然是使用双指针法。考虑构建一个指针，此指针需要有以下性质：此指针和 slow 一起向前走 a 步后，两者在入口节点重合。那么从哪里走到入口节点需要 a 步？答案是链表头节点head。
>
> 1. 令 `fast` 重新指向链表头部节点。此时 f=0，s=nb 。（代码里是直接移动了head，可以复用fast或重新创建个ptr = head）
> 2. `slow` 和 `fast` 同时每轮向前走 1 步。
> 3. 当 fast 指针走到 f=a 步时，slow 指针走到 s=a+nb 步。此时两指针重合，并同时指向链表环入口，返回 slow 指向的节点即可。
>

## 144 二叉树的前序遍历

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 递归版本
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if not root:
                return
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)
            
        res = []
        dfs(root)
        return res
    
# 迭代版本：写法类似BFS，但要注意这里用的是stack，前序遍历是DLR，所以入栈要先右子树再左子树
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
```

## 145 二叉树的后序遍历

给你一棵二叉树的根节点 `root` ，返回其节点值的 **后序遍历** 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            dfs(root.right)
            res.append(root.val)
        
        res = []
        dfs(root)
        return res
```

## 148 排序链表

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# 自顶向下归并排序
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def sortFunc(head, tail):
            # 空链表
            if not head:
                return head
            # 只有一个节点
            if head.next == tail:
                head.next = None
                return head
            # 快慢指针找链表中点
            slow = fast = head
            while fast != tail: # 这里只能用fast!=tail来判定，不能用fast and fast.next，因为递归过程中fast到子链表尾部时并不为空
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))
        
        # 合并两个有序链表，递归或迭代均可
        def merge(head1, head2):
            if not head1:
                return head2
            if not head2:
                return head1
            if head1.val <= head2.val:
                head1.next = merge(head1.next, head2)
                return head1
            else:
                head2.next = merge(head1, head2.next)
                return head2
        
        return sortFunc(head, None)
```

> 「147. 对链表进行插入排序」要求使用插入排序的方法对链表进行排序，插入排序的时间复杂度是 O(n2)，其中 n 是链表的长度。这道题考虑时间复杂度更低的排序算法。题目的进阶问题要求达到 O(nlog⁡n) 的时间复杂度和 O(1) 的空间复杂度，时间复杂度是 O(nlog⁡n) 的排序算法包括归并排序、堆排序和快速排序（快速排序的最差时间复杂度是 O(n^2)，其中最适合链表的排序算法是归并排序。
>
> 归并排序基于分治算法。最容易想到的实现方式是自顶向下的递归实现，考虑到递归调用的栈空间，自顶向下归并排序的空间复杂度是 O(log⁡n)。如果要达到 O(1) 的空间复杂度，则需要使用自底向上的实现方式。
>
> 对链表自顶向下归并排序的过程如下：
>
> 1. 找到链表的中点，以中点为分界，将链表拆分成两个子链表。寻找链表的中点可以使用快慢指针的做法，快指针每次移动 2 步，慢指针每次移动 1 步，当快指针到达链表末尾时，慢指针指向的链表节点即为链表的中点。
> 2. 对两个子链表分别排序。
> 3. 将两个排序后的子链表合并，得到完整的排序后的链表。可以使用[21. 合并两个有序链表]的做法，将两个有序的子链表进行合并。
>
> 上述过程可以通过递归实现。递归的终止条件是链表的节点个数小于或等于 1，即当链表为空或者链表只包含 1 个节点时，不需要对链表进行拆分和排序。

## 151 反转字符串中的单词

给你一个字符串 `s` ，请你反转字符串中 **单词** 的顺序。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

返回 **单词** 顺序颠倒且 **单词** 之间用单个空格连接的结果字符串。

**注意：**输入字符串 `s`中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 两次翻转：先翻转所有字符，再逐词翻转
        s = s.strip()
        reversed_s = s[::-1]
        res = []
        for word in reversed_s.split():
            res.append(word[::-1])
        return ' '.join(res)
    
# 一行流
class Solution:
    def reverseWords(self, s: str) -> str:
        # return ' '.join(reversed(s.strip().split()))
        return ' '.join(s.strip().split()[::-1])

# 双指针，倒序遍历
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        i, j = len(s)-1, len(s)-1
        res = []
        while i >= 0:
            # 找单词前的第一个空格
            while i >= 0 and s[i] != ' ':
                i -= 1
            res.append(s[i+1: j+1])
            # 跳过单词间空格，到前一个单词的最后一个字母
            while i >= 0 and s[i] == ' ':
                i -= 1
            j = i
        return ' '.join(res)
```

> 双指针倒序遍历：倒序遍历字符串 *s* ，记录单词左右索引边界 *i* , *j* ；每确定一个单词的边界，则将其添加至单词列表 *res* ；最终，将单词列表拼接为字符串，并返回即可

## 152 乘积最大子数组

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

**子数组** 是数组的连续子序列。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        dp_max, dp_min = [0] * n, [0] * n
        dp_max[0] = dp_min[0] = nums[0]
        for i in range(1, n):
            dp_max[i] = max(nums[i], dp_max[i-1]*nums[i], dp_min[i-1]*nums[i])
            dp_min[i] = min(nums[i], dp_max[i-1]*nums[i], dp_min[i-1]*nums[i])
        return max(dp_max)
```

动态规划，递推思路类似于53题，但这里是乘积，而不是求和

> 由于存在负数，那么会导致最大的变最小的，最小的变最大的。因此还需要维护当前最小值
>
> 考虑当前位置如果是一个负数的话，那么我们希望以它前一个位置结尾的某个段的积也是个负数，这样就可以负负得正，并且我们希望这个积尽可能「负得更多」，即尽可能小。如果当前位置是一个正数的话，我们更希望以它前一个位置结尾的某个段的积也是个正数，并且希望它尽可能地大。于是这里我们可以再维护一个 fmin⁡(i)，它表示以第 i 个元素结尾的乘积最小子数组的乘积
>

## 160 相交链表

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 做法一：哈希集合存储链表节点，时间复杂度O(m+n)，空间复杂度O(m)
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        hash_table = set()
        while headA:
            hash_table.add(headA)
            headA = headA.next
        while headB:
            if headB in hash_table:
                return headB
            headB = headB.next
        return None

# 双指针，时间复杂度O(m+n)，空间复杂度O(1)
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None
        pA, pB = headA, headB
        while pA or pB:
            if not pA:
                pA = headB
            if not pB:
                pB = headA
            if pA == pB:
                return pA
            pA = pA.next
            pB = pB.next
        return None
```

> 当链表 headA 和 headB 都不为空时，创建两个指针 pA 和 pB，初始时分别指向两个链表的头节点 headA 和 headB，然后将两个指针依次遍历两个链表的每个节点。具体做法如下：
>
> - 每步操作需要同时更新指针 pA 和 pB。
> - 如果指针 pA 不为空，则将指针 pA 移到下一个节点；如果指针 pB 不为空，则将指针 pB 移到下一个节点。
> - 如果指针 pA 为空，则将指针 pA 移到链表 headB 的头节点；如果指针 pB 为空，则将指针 pB 移到链表 headA 的头节点。
> - 当指针 pA 和 pB 指向同一个节点或者都为空时，返回它们指向的节点或者 null。

## 169 多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```python
# 做法一：哈希表，统计每个元素出现的次数，时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        hash_table = {}
        for num in nums:
            if num in hash_table:
                hash_table[num] += 1
                if hash_table[num] > n // 2:
                    return num
            else:
                hash_table[num] = 1
                
# 做法二：排序
# 如果将数组 nums 中的所有元素按照单调递增或单调递减的顺序排序，那么下标为 ⌊n/2⌋ 的元素一定是众数
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        return nums[n//2]
    
# 做法三：分治，如果数 a 是数组 nums 的众数，如果我们将 nums 分成两部分，那么 a 必定是至少一部分的众数。
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def majority_element(low, high):
            # base case
            if low == high:
                return nums[low]
            
            # 分别计算左右子区间的众数
            mid = low + (high - low) // 2
            left = majority_element(low, mid)
            right = majority_element(mid+1, high)
            
            # 左右子区间众数相同直接返回
            if left == right:
                return left
            # 左右子区间众数不同需要比较两者在整个区间内出现的次数来决定最终众数
            left_count = sum([1 for i in range(low, high+1) if nums[i] == left])
            right_count = sum([1 for i in range(low, high+1) if nums[i] == right])
            return left if left_count > right_count else right
        
        return majority_element(0, len(nums)-1)
```

## 198 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[n-1]
```

动态规划

> *dp*[*i*] 表示前 *i* 间房屋能偷窃到的最高总金额
>
> dp[0]=nums[0]，只有一间房屋，则偷窃该房屋
>
> dp[1]=max(nums[0],nums[1])，只有两间房屋，选择其中金额较高的房屋进行偷窃
>
> 如果房屋数量大于两间，应该如何计算能够偷窃到的最高总金额呢？对于第 i (i>2) 间房屋，有两个选项：
>
> 1. 偷窃第 *i* 间房屋，那么就不能偷窃第 i−1 间房屋，偷窃总金额为前 i−2 间房屋的最高总金额与第 i 间房屋的金额之和。
> 2. 不偷窃第 *i* 间房屋，偷窃总金额为前 i-1 间房屋的最高总金额。
>
> 在两个选项中选择偷窃总金额较大的选项，该选项对应的偷窃总金额即为前 k 间房屋能偷窃到的最高总金额。
>
> *dp*[*i*]=max(*dp*[*i*−2]+*nums*[*i*],*dp*[*i*−1])

## 200 岛屿数量

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def check(grid, i, j):
            if 0 <= i <= len(grid)-1 and 0 <= j <= len(grid[0])-1:
                return True
            else:
                return False

        def dfs(grid, i, j):
            grid[i][j] = '0'
            
            for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if check(grid, k[0], k[1]) and grid[k[0]][k[1]] == '1':
                    dfs(grid, k[0], k[1])
                    
        m = len(grid)
        n = len(grid[0])
        count = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
        return count
```

dfs

## 206 反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 空链表
        if not head:
            return None
        # 单个节点
        if not head.next:
            return head
        
        prev, curr = head, head.next
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        # pre初始化为head而不是None，导致反转是从第二个节点head.next开始的，因此反转后head仍然指向head.next而不是None，出现死循环
        head.next = None
        return prev
    
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        return prev
```

> 在遍历链表时，将当前节点的 next 指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头引用。
>

## 207 课程表

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 入度数组，邻接表
        indegree = [0] * numCourses
        map = {} # 课号: 依赖这门课的后续课列表
        for i in prerequisites:
            indegree[i[0]] += 1
            if i[1] in map.keys():
                map[i[1]].append(i[0])
            else:
                map.update({i[1]: [i[0]]})
        
        queue = []
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
        
        count = 0
        while queue:
            course = queue.pop(0)
            count += 1
            # 如果有依赖此课程的后续课程则更新入度
            if course in map.keys():
                for i in map[course]:
                    indegree[i] -= 1
                    # 后续课程除去当前课程无其他依赖课程则丢入队列
                    if indegree[i] == 0:
                        queue.append(i)
        return count == numCourses
```

拓扑排序，bfs

> 1. 根据依赖关系，构建邻接表、入度数组。
> 2. 选取入度为 0 的数据，根据邻接表，减小依赖它的数据的入度。
> 3. 找出入度变为 0 的数据，重复第 2 步。
> 4. 直至所有数据的入度为 0，得到排序，如果还有数据的入度不为 0，说明图中存在环。

## 210 课程表 II

现在你总共有 `numCourses` 门课需要选，记为 `0` 到 `numCourses - 1`。给你一个数组 `prerequisites` ，其中 `prerequisites[i] = [ai, bi]` ，表示在选修课程 `ai` 前 **必须** 先选修 `bi` 。

- 例如，想要学习课程 `0` ，你需要先完成课程 `1` ，我们用一个匹配来表示：`[0,1]` 。

返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 **任意一种** 就可以了。如果不可能完成所有课程，返回 **一个空数组** 。

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = [0] * numCourses
        map = {}
        for i in prerequisites:
            indegree[i[0]] += 1
            if i[1] in map.keys():
                map[i[1]].append(i[0])
            else:
                map.update({i[1]: [i[0]]})
        
        queue = []
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
        
        res = []
        while queue:
            course = queue.pop(0)
            res.append(course)
            if course in map.keys():
                for i in map[course]:
                    indegree[i] -= 1
                    if indegree[i] == 0:
                        queue.append(i)
        if len(res) != numCourses:
            return []
        return res
```

拓扑排序，bfs，相比于207多了一步用res记录节点

## 213 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你**在不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])
        
        # 偷第一间
        dp_first = [0] * n
        dp_first[0] = nums[0]
        dp_first[1] = max(nums[0], nums[1])
        
        # 偷最后一间
        dp_last = [0] * n
        dp_last[1] = nums[1]
        dp_last[2] = max(nums[1], nums[2])

        for i in range(2, n-1):
            dp_first[i] = max(dp_first[i-2]+nums[i], dp_first[i-1])
        for i in range(3, n):
            dp_last[i] = max(dp_last[i-2]+nums[i], dp_last[i-1])
        return max(dp_first[n-2], dp_last[n-1])
```

> 注意到当房屋数量不超过两间时，最多只能偷窃一间房屋，因此不需要考虑首尾相连的问题。如果房屋数量大于两间，就必须考虑首尾相连的问题，第一间房屋和最后一间房屋不能同时偷窃。
>
> 如何才能保证第一间房屋和最后一间房屋不同时偷窃呢？如果偷窃了第一间房屋，则不能偷窃最后一间房屋，因此偷窃房屋的范围是第一间房屋到最后第二间房屋；如果偷窃了最后一间房屋，则不能偷窃第一间房屋，因此偷窃房屋的范围是第二间房屋到最后一间房屋。
>
> 假设数组 nums 的长度为 n。如果不偷窃最后一间房屋，则偷窃房屋的下标范围是 [0,n−2]；如果不偷窃第一间房屋，则偷窃房屋的下标范围是 [1,n−1]。在确定偷窃房屋的下标范围之后，即可用第 198 题的方法解决。对于两段下标范围分别计算可以偷窃到的最高总金额，其中的最大值即为在 n 间房屋中可以偷窃到的最高总金额。
>
> 其实就是把环拆成两个队列，一个是从0到n-2，另一个是从1到n-1，然后返回两个结果最大的。

## 215 数组中的第 K 个最大元素

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```python
# 做法一：暴力，时间复杂度O(nlogn)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums)-k]
    
# 做法二：快速选择，改进快排
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quicksort(nums, l, r, target):
            if l >= r:
                return
            
            pivot = nums[l]
            i, j = l, r
            while i < j:
                # 改为降序排序
                while i < j and nums[j] <= pivot:
                    j -= 1
                while i < j and nums[i] >= pivot:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l] = nums[i]
            nums[i] = pivot
            
            # 判断划分位置
            if i < k-1:
                quicksort(nums, i+1, r, target)
            else:
                quicksort(nums, l, i-1, target)

        # 第k大就是升序排序后的n-k位置元素，也就是降序排序后的k-1位置 
        quicksort(nums, 0, len(nums)-1, k-1)
        return nums[k-1]
```

> 我们可以改进快速排序算法来解决这个问题：在分解的过程当中，我们会对子数组进行划分，如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。这样就可以把原来递归两个区间变成只递归一个区间，提高了时间效率。这就是「快速选择」算法。
>

## 221 最大正方形

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        side = 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    side = max(side, dp[i][j])
        return side ** 2
```

> dp(i,j) 表示以 (i,j) 为右下角，且只包含 1 的正方形的边长最大值。如果我们能计算出所有 dp(i,j) 的值，那么其中的最大值即为矩阵中只包含 1 的正方形的边长最大值，其平方即为最大正方形的面积。
>
> 对于每个位置 (i,j)，检查在矩阵中该位置的值：如果该位置的值是 0，则 dp(i,j)=0，因为当前位置不可能在由 1 组成的正方形中；
>
> 如果该位置的值是 1，则 dp(i,j) 的值由其上方、左方和左上方的三个相邻位置的 dp 值决定。具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1，状态转移方程dp(i,j)=min(dp(i−1,j),dp(i−1,j−1),dp(i,j−1))+1
>
> 此外，还需要考虑边界条件。如果 i 和 j 中至少有一个为 0，则以位置 (i,j)(i, j)(i,j) 为右下角的最大正方形的边长只能是 1，因此 dp(i,j)=1。
>
> 当我们判断以某个点为正方形右下角时最大的正方形时，那它的上方，左方和左上方三个点也一定是某个正方形的右下角，否则该点为右下角的正方形最大就是它自己了。这是定性的判断，那具体的最大正方形边长呢？我们知道，该点为右下角的正方形的最大边长，最多比它的上方，左方和左上方为右下角的正方形的边长多1，最好的情况是是它的上方，左方和左上方为右下角的正方形的大小都一样的，这样加上该点就可以构成一个更大的正方形。 但如果它的上方，左方和左上方为右下角的正方形的大小不一样，合起来就会缺了某个角落，这时候只能取那三个正方形中最小的正方形的边长加1了。假设dpi表示以i,j为右下角的正方形的最大边长，则有 `dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1` 当然，如果这个点在原矩阵中本身就是0的话，那dp[i]肯定就是0了。

## 226 翻转二叉树

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 思路一：先递归到叶节点，返回时交换左右节点的指向，速度更快
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        left = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(left)
        return root

# 思路二：先交换当前节点的左右子节点，然后分别递归左右子树
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

## 234 回文链表

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# 做法一：将值复制到数组中后用双指针法
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        
        # return vals == vals[::-1]
        l, r = 0, len(vals)-1
        while l < r:
            if vals[l] != vals[r]:
                return False
            l += 1
            r -= 1
        return True

# 做法二：快慢指针，将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # 快慢指针找链表中点
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 反转后半部分链表
        prev, curr = None, slow
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        
        # 双指针判断回文
        while prev:
            if head.val != prev.val:
                return False
            head = head.next
            prev = prev.next
        return True
```

## 235 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == p or root == q:
            return root
        
        val = root.val
        p_val = p.val
        q_val = q.val
        if p_val < val and q_val < val:
            left = self.lowestCommonAncestor(root.left, p, q)
            return left
        elif p_val > val and q_val > val:
            right = self.lowestCommonAncestor(root.right, p, q)
            return right
        else:
            return root

# 合并“返回当前节点”情况的写法
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        val = root.val
        p_val = p.val
        q_val = q.val
        if p_val < val and q_val < val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p_val > val and q_val > val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
```

> 分类讨论：
>
> 1. 当前节点是p；当前节点是q：返回当前节点
> 2. p和q分别在左右子树：返回当前节点
> 3. p和q都在左子树：返回递归左子树的结果
> 4. p和q都在右子树：返回递归右子树的结果
> 5. p和q都不在左右子树：返回空节点
>
> 与236不同的是，本题不需要讨论“当前节点是空节点”，因为BST是有序的，根据p和q的值直接就去合适的左右子树遍历节点，找到p和q就返回，不会遍历到空节点，而236因为不知道p和q的位置，所以可能遍历到空节点（比如，p和q都在左子树，遍历右子树时就会一直向下遍历到空节点）

## 236 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif left:
            return left
        elif right:
            return right
        else:
            return None
```

> 分类讨论：
>
> 1. 当前节点是空节点；当前节点是p；当前节点是q：返回当前节点
> 2. p和q分别在左右子树：返回当前节点
> 3. p和q都在左子树：返回递归左子树的结果
> 4. p和q都在右子树：返回递归右子树的结果
> 5. p和q都不在左右子树：返回空节点

## 238 除自身以外数组的乘积

给你一个整数数组 `nums`，返回 *数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积* 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请**不要使用除法，**且在 `O(n)` 时间复杂度内完成此题。

```python
# 做法一：左右乘积列表，前缀和（积）
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        # 从前往后前缀积
        p_front = [nums[0]] * n
        for i in range(1, n):
            p_front[i] = p_front[i-1] * nums[i]
        # 从后往前后缀积
        p_back = [nums[n-1]] * n
        for i in reversed(range(n-1)):
            p_back[i] = p_back[i+1] * nums[i]
        
        answer = [0] * n
        for i in range(n):
            if i == 0:
                answer[i] = p_back[i+1]
            elif i == n-1:
                answer[i] = p_front[i-1]
            else:
                answer[i] = p_front[i-1] * p_back[i+1]
        return answer
```

## 240 搜索二维矩阵 II

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

```python
# 做法一：暴力遍历
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == target:
                    return True
        return False
    
# 做法二：二分查找
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def binarysearch(nums, target):
            l, r = 0, len(nums)-1
            while l <= r:
                mid = l + (r - l) // 2
                if nums[mid] == target:
                    return True
                elif nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
            return False

        for row in matrix:
            if binarysearch(row, target):
                return True
        return False
    
# 做法三：Z字形查找
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        i, j = 0, n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                i += 1
            else:
                j -= 1
        return False
```

> 我们可以从矩阵 matrix 的右上角 (0,n−1) 进行搜索。在每一步的搜索过程中，如果我们位于位置 (x,y)，那么我们希望在以 matrix 的左下角为左下角、以 (x,y) 为右上角的矩阵中进行搜索，即行的范围为 [x, m - 1]，列的范围为 [0, y]：如果 matrix[x,y]=target，说明搜索完成；如果 matrix[x,y]>target，由于每一列的元素都是升序排列的，那么在当前的搜索矩阵中，所有位于第 y 列的元素都是严格大于 target 的，因此我们可以将它们全部忽略，即将 y 减少 1；如果 matrix[x,y]<target，由于每一行的元素都是升序排列的，那么在当前的搜索矩阵中，所有位于第 x 行的元素都是严格小于 target 的，因此我们可以将它们全部忽略，即将 x 增加 1。在搜索的过程中，如果我们超出了矩阵的边界，那么说明矩阵中不存在 target。
>
> 二分查找每次搜索可以排除半行或半列的元素，Z字形查找每次搜索可以排除一行或一列的元素。

## 263 丑数

**丑数** 就是只包含质因数 `2`、`3` 和 `5` 的正整数。

给你一个整数 `n` ，请你判断 `n` 是否为 **丑数** 。如果是，返回 `true` ；否则，返回 `false` 。

```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False
        
        factors = [2, 3, 5]
        for factor in factors:
            while n % factor == 0:
                n //= factor
        return n == 1
```

## 279 完全平方数

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # 计算所有小于等于n的完全平方数，即物品数组
        nums = []
        k = 1
        while k*k <= n:
            nums.append(k*k)
            k += 1
        
        target = n
        dp = [target+1] * (target+1)
        dp[0] = 0
        for i in range(len(nums)):
            for j in range(nums[i], target+1):
                dp[j] = min(dp[j], dp[j-nums[i]]+1)
        return dp[target]
```

和322题几乎一模一样

## 283 移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ptr = 0
        for curr in range(len(nums)):
            if nums[curr] != 0:
                nums[ptr], nums[curr] = nums[curr], nums[ptr]
                ptr += 1
```

双指针，同75题的做法二。使用一个指针 ptr 表示「头部」的范围，ptr 中存储了一个整数，表示数组 nums 从位置 0 到位置 ptr−1 都属于「头部」。ptr 的初始值为 0，表示还没有数处于「头部」。

## 287 寻找重复数

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

```python
# 哈希表，空间复杂度O(n)，不符合要求
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        hashtable = set()
        for num in nums:
            if num in hashtable:
                return num
            else:
                hashtable.add(num)
        return -1
    
# 排序+双指针，空间复杂度O(1)，修改了数组，不符合要求
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        if n == 1:
            return nums[0]
        
        i, j = 0, 1
        while j < n:
            if nums[i] == nums[j]:
                return nums[i]
            else:
                i += 1
                j += 1
        return -1
    
# 快慢指针，符合要求
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = fast = 0
        # 一定存在重复的整数，即链表一定存在环
        # 按链表的写法也可以while nums[fast] and nums[nums[fast]]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                head = 0
                while slow != head:
                    slow = nums[slow]
                    head = nums[head]
                return slow
```

> 使用环形链表II的方法解题，使用 142 题的思想来解决此题的关键是要理解如何将输入的数组看作为链表。 首先明确前提，整数的数组 nums 中的数字范围是 [1,n]。考虑一下两种情况：
>
> 如果数组中没有重复的数，以数组 [1,3,4,2]为例，我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)， 其映射关系 n->f(n)为： 0->1 1->3 2->4 3->2 我们从下标为 0 出发，根据 f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样可以产生一个类似链表一样的序列。 0->1->3->2->4->null
>
> 如果数组中有重复的数，以数组 [1,3,4,2,2] 为例,我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)， 其映射关系 n->f(n) 为： 0->1 1->3 2->4 3->2 4->2 同样的，我们从下标为 0 出发，根据 f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推产生一个类似链表一样的序列。 0->1->3->2->4->2->4->2->…… 这里 2->4 是一个循环。
>
> 从理论上讲，数组中如果有重复的数，那么就会产生多对一的映射，这样，形成的链表就一定会有环路了，
>
> 综上：数组中有一个重复的整数 <=> 链表中存在环；找到数组中的重复整数 <=> 找到链表的环入口。至此，问题转换为 142 题。那么针对此题，快慢指针该如何走呢。根据上述数组转链表的映射关系，可推出 142 题中慢指针走一步 slow = slow.next => 本题 slow = nums[slow]；142 题中快指针走两步 fast = fast.next.next ==> 本题 fast = nums[nums[fast]]。其他的部分和142题一样。
>

## 300 最长递增子序列

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        # dp[i] = max(dp[j]) + 1, 0 <= j < i, nums[j] < nums[i]
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```

>  dp[i] 为考虑前 i 个元素，以第 i 个数字结尾的最长递增子序列的长度，注意 nums[i] 必须被选取。我们从小到大计算 dp 数组的值，在计算 dp[i] 之前，我们已经计算出 dp[0…i−1] 的值，则状态转移方程为：dp[i]=max⁡(dp[j])+1,其中 0≤j<i 且 num[j]<num[i]，即考虑往 dp[0…i−1] 中最长的递增子序列后面再加一个 nums[i]。由于 dp[j] 代表 nums[0…j] 中以 nums[j] 结尾的最长递增子序列，所以如果能从 dp[j] 这个状态转移过来，那么 nums[i] 必然要大于 nums[j]，才能将 nums[i] 放在 nums[j] 后面以形成更长的递增子序列。最后，整个数组的最长递增子序列即所有 dp[i] 中的最大值。

## 309 买卖股票的最佳时机含冷冻期

给定一个整数数组`prices`，其中第 `prices[i]` 表示第 `i` 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        f = [[0] * 3 for _ in range(n)]
        f[0][0] = -prices[0]

        for i in range(1, n):
            f[i][0] = max(f[i-1][0], f[i-1][2]-prices[i])
            f[i][1] = f[i-1][0] + prices[i]
            f[i][2] = max(f[i-1][1], f[i-1][2])
        return max(f[n-1][1], f[n-1][2])
```

> f[i] 表示第 i 天结束之后的「累计最大收益」。根据题目描述，由于我们最多只能同时买入（持有）一支股票，并且卖出股票后有冷冻期的限制，因此我们会有三种不同的状态：
>
> 我们目前持有一支股票，对应的「累计最大收益」记为 f\[i][0]；
>
> 我们目前不持有任何股票，并且处于冷冻期中，对应的「累计最大收益」记为 f\[i][1]；
>
> 我们目前不持有任何股票，并且不处于冷冻期中，对应的「累计最大收益」记为 f\[i][2]
>
> 这里的「处于冷冻期」指的是在第 i 天结束之后的状态。也就是说：如果第 i 天结束之后处于冷冻期，那么第 i+1 天无法买入股票。
>
> 对于 f\[i][0]，我们目前持有的这一支股票可以是在第 i−1 天就已经持有的，对应的状态为 f\[i−1][0]；或者是第 i 天买入的，那么第 i−1 天就不能持有股票并且不处于冷冻期中，对应的状态为 f\[i−1][2] 加上买入股票的负收益 prices[i]。因此状态转移方程为：f\[i][0]=max⁡(f\[i−1][0],f\[i−1][2]−prices[i])。
>
> 对于 f\[i][1]，我们在第 i 天结束之后处于冷冻期的原因是在当天卖出了股票，那么说明在第 i−1 天时我们必须持有一支股票，对应的状态为 f\[i−1][0] 加上卖出股票的正收益 prices[i]。因此状态转移方程为：f\[i][1]=f\[i−1][0]+prices[i]。
>
> 对于 f\[i][2]，我们在第 i 天结束之后不持有任何股票并且不处于冷冻期，说明当天没有进行任何操作，即第 i−1 天时不持有任何股票：如果处于冷冻期，对应的状态为 f\[i−1][1]；如果不处于冷冻期，对应的状态为 f\[i−1][2]。因此状态转移方程为：f\[i][2]=max⁡(f\[i−1][1],f\[i−1][2])。
>
> 这样我们就得到了所有的状态转移方程。如果一共有 n 天，那么最终的答案即为：max⁡(f\[n−1][0],f\[n−1][1],f\[n−1][2])，注意到如果在最后一天（第 n−1 天）结束之后，手上仍然持有股票，那么显然是没有任何意义的。因此更加精确地，最终的答案实际上是 max⁡(f\[n−1][1],f\[n−1][2])。
>
> 初始化：f\[0][0]=−prices[0]，f\[0][1]=0，f\[0][2]=0。在第 0 天时，如果持有股票，那么只能是在第 0 天买入的，对应负收益 −prices[0]；如果不持有股票，那么收益为零。
>

## 322 零钱兑换

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        # 初始化为正无穷或不可能取到的较大值，防止影响递推的min()
        dp = [amount+1] * (amount+1)
        # 总金额0所需数量为0
        dp[0] = 0

        for i in range(n):
            for j in range(coins[i], amount+1):
                dp[j] = min(dp[j], dp[j-coins[i]]+1)
        return dp[amount] if dp[amount] < amount+1 else -1
```

完全背包，注意初始化和最后输出前的判断

## 337 打家劫舍 III

小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root` 。

除了 `root` 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 **两个直接相连的房子在同一天晚上被打劫** ，房屋将自动报警。

给定二叉树的 `root` 。返回 ***在不触动警报的情况下** ，小偷能够盗取的最高金额* 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            # 递归边界
            if not root:
                return 0, 0
            
            l_rob, l_not_rob = dfs(root.left)
            r_rob, r_not_rob = dfs(root.right)
            # 偷当前节点
            rob = root.val + l_not_rob + r_not_rob
            # 不偷当前节点
            not_rob = max(l_rob, l_not_rob) + max(r_rob, r_not_rob)
            return rob, not_rob

        return max(dfs(root))
```

树形dp

> 简化一下这个问题：一棵二叉树，树上的每个点都有对应的权值，每个点有两种状态（选中和不选中），问在不能同时选中有父子关系的点的情况下，能选中的点的最大权值和是多少。
>
> 若当前节点被选中时，其左右子节点都不能被选中，故当前节点被选中情况下子树上被选中点的最大权值和：当前节点值+不选左右子节点的值；若当前节点不被选中时，其左右子节点可选可不选，取较大的情况。最后遍历顺序应该用后序，自底向上统计，保证在访问当前节点时，它的左右子节点已经被计算过了。
>

## 338 比特位计数

给你一个整数 `n` ，对于 `0 <= i <= n` 中的每个 `i` ，计算其二进制表示中 **`1` 的个数** ，返回一个长度为 `n + 1` 的数组 `ans` 作为答案。

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        def countOnes(x):
            count = 0
            while x > 0:
                x &= x-1
                count += 1
            return count
        
        res = []
        for i in range(n+1):
            res.append(countOnes(i))
        return res
```

> Brian Kernighan 算法：
>
> 最直观的做法是对从 0 到 n 的每个整数直接计算「一比特数」。每个 int 型的数都可以用 32 位二进制数表示，只要遍历其二进制表示的每一位即可得到 1 的数目。
>
> 利用 Brian Kernighan 算法，可以在一定程度上进一步提升计算速度。Brian Kernighan 算法的原理是：对于任意整数 x，令 x=x & (x−1)，该运算将 x 的二进制表示的最后一个 1 变成 0。因此，对 x 重复该操作，直到 x 变成 0，则操作次数即为 x 的「一比特数」。
>
> 对于给定的 n，计算从 0 到 n 的每个整数的「一比特数」的时间都不会超过 O(log⁡n)，因此总时间复杂度为 O(nlog⁡n)。

## 416 分割等和子集

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        # 和为奇数，不可能二等分
        if s % 2 != 0:
            return False
        
        # 01背包模板，背包容量s//2，len(nums)个物品，第i个物品的体积nums[i]，价值也是nums[i]
        target = s // 2
        n = len(nums)
        dp = [[0] * (target+1) for _ in range(n)]
        for j in range(target+1):
            dp[0][j] = 0 if nums[0] > j else nums[0]
        
        for i in range(1, n):
            for j in range(target+1):
                if nums[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-nums[i]]+nums[i])
        return dp[n-1][target] == target
    
# 一维dp数组做法
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        # 和为奇数，不可能二等分
        if s % 2 != 0:
            return False
        
        target = s // 2
        n = len(nums)
        dp = [0] * (target+1)
        
        for i in range(n):
            for j in reversed(range(nums[i], target+1)):
                dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
        return dp[target] == target

# 一维dp数组：直接做法
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s % 2 != 0:
            return False
        
        target = s // 2
        n = len(nums)
        dp = [False] * (target+1)
        dp[0] = True

        for i in range(1, n):
            for j in reversed(range(nums[i], target+1)):
                dp[j] = dp[j] or dp[j-nums[i]]
        return dp[target]
```

> 这道题可以换一种表述：给定一个只包含正整数的非空数组 nums，判断是否可以从数组中选出一些数字，使得这些数字的和等于整个数组的元素和的一半。因此这个问题可以转换成「0−1 背包问题」。这道题与传统的「0−1 背包问题」的区别在于，传统的「0−1 背包问题」要求选取的物品的重量之和不能超过背包的总容量，这道题则要求选取的数字的和恰好等于整个数组的元素和的一半。
>
> 直接做法：dp\[i][j]代表考虑前 i 个数值，其选择数字总和是否恰好为 j，此时dp数组中存储的是「布尔类型」的动规值。新转移方程代表的意思为：dp\[i][j]想要为真 (考虑前 i 个数值，选择的数字总和恰好为 j) 。需要满足以下两种方案，至少一种为 ：不选第 i 件物品，选择的数字总和恰好为 j；选第 i 件物品，选择的数字总和恰好为 j。初始化时，增加一个「不考虑任何物品」的情况讨论，之前我们的状态定义是 代表考虑下标为 i 之前的所有物品。现在我们可以加入「不考虑任何物品」的情况，也就是将「物品编号」从 0 开始调整为从 1 开始，遍历物品从1开始（其实也是为了后面能推出True而设立的dummy值，不然全初始化为False，递推公式又是or，不可能推出True，这种技巧称为构造有效值）
>
> 我们可以通过将一个背包问题的「状态定义」从「最多不超过 XX 容量」修改为「背包容量恰好为 XX」，同时再把「有效值构造」出来，也即是将「物品下标调整为从 1 开始，设置 dp[0] 为初始值」。这其实是另外一类「背包问题」，它不对应「价值最大化」，对应的是「能否取得最大/特定价值」。这样的「背包问题」同样具有普遍性。

## 437 路径总和 III

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # 搜索以root为根的所有向下的和为target的路径
        def dfs(root, target):
            if not root:
                return 0
            
            count = 0
            if root.val == target:
                count += 1
            return count + dfs(root.left, target-root.val) + dfs(root.right, target-root.val)
        
        if not root:
            return 0
        # 搜索所有节点
        return dfs(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)
```

> 我们首先想到的解法是穷举所有的可能，我们访问每一个节点 node，检测以 node 为起始节点且向下延深的路径有多少种。我们递归遍历每一个节点的所有可能的路径，然后将这些路径数目加起来即为返回结果。
>
> 我们首先定义 rootSum(p,val) 表示以节点 p 为起点向下且满足路径总和为 val 的路径数目。我们对二叉树上每个节点 p 求出 rootSum(p,targetSum)，然后对这些路径数目求和即为返回结果。
>
> 我们对节点 p 求 rootSum(p,targetSum) 时，以当前节点 p 为目标路径的起点递归向下进行搜索。假设当前的节点 p 的值为 val，我们对左子树和右子树进行递归搜索，对节点 p 的左孩子节点 pl 求出 rootSum(pl,targetSum−val)，以及对右孩子节点 pr 求出 rootSum(pr,targetSum−val)。节点 p 的 rootSum(p,targetSum) 即等于 rootSum(pl,targetSum−val) 与 rootSum(pr,targetSum−val) 之和，同时我们还需要判断一下当前节点 p 的值是否刚好等于 targetSum。
>
> 我们采用递归遍历二叉树的每个节点 p，对节点 p 求 rootSum(p,val)，然后将每个节点所有求的值进行相加求和返回。
>

## 448 找到所有数组中消失的数字

给你一个含 `n` 个整数的数组 `nums` ，其中 `nums[i]` 在区间 `[1, n]` 内。请你找出所有在 `[1, n]` 范围内但没有出现在 `nums` 中的数字，并以数组的形式返回结果。

```python
# Python内置函数，求集合差集
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        nums_set = set(nums)
        res = set([i for i in range(1, len(nums)+1)])
        return list(res-nums_set)

# 双重循环，超时
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = []
        for i in range(1, n+1):
            if i not in nums:
                res.append(i)
        return res
    
# 辅助数组，空间复杂度O(n)
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        # 这里创建长度为n的数组，因为nums可能存在数字n，需要对res索引减一，最后加回来
        # 也可以直接创建n+1长度的数组，将索引0的值初始化为1，res = [1]+ [0] * n
        res = [0] * n
        for num in nums:
            if res[num-1] == 0:
                res[num-1] = 1
        return [i+1 for i, num in enumerate(res) if num == 0]
    
# 原地修改，空间复杂度O(1)
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for num in nums:
            x = (num - 1) % n
            nums[x] += n
        res = [i+1 for i, num in enumerate(nums) if num <= n]
        return res
```

> 我们可以用一个哈希表记录数组 nums 中的数字，由于数字范围均在 [1,n] 中，记录数字后我们再利用哈希表检查 [1,n] 中的每一个数是否出现，从而找到缺失的数字。由于数字范围均在 [1,n] 中，我们也可以用一个长度为 n 的数组来代替哈希表。
>
> 这一做法的空间复杂度是 O(n) 的。我们的目标是优化空间复杂度到 O(1)。
>
> 注意到 nums 的长度恰好也为 nnn，能否让 nums 充当哈希表呢？由于 nums 的数字范围均在 [1,n] 中，我们可以利用这一范围之外的数字，来表达「是否存在」的含义。具体来说，遍历 nums，每遇到一个数 x，就让 nums[x−1] 增加 n。由于 nums 中所有数均在 [1,n] 中，增加以后，这些数必然大于 n。最后我们遍历 nums，若 nums[i] 未大于 n，就说明没有遇到过数 i+1。这样我们就找到了缺失的数字。注意，当我们遍历到某个位置时，其中的数可能已经被增加过，因此需要对 n 取模来还原出它本来的值。
>

## 461 汉明距离

两个整数之间的 [汉明距离](https://baike.baidu.com/item/汉明距离) 指的是这两个数字对应二进制位不同的位置的数目。

给你两个整数 `x` 和 `y`，计算并返回它们之间的汉明距离。

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        s = x ^ y
        count_one = 0
        while s > 0:
            s &= s-1
            count_one += 1
        return count_one
```

计算 *x* 和 *y* 之间的汉明距离，可以先计算 *x*⊕*y*，然后统计结果中等于 1 的位数（相异为真）。统计一比特数时用的和338题一样的Brian Kernighan 算法。

## 463 岛屿的周长

给定一个 `row x col` 的二维网格地图 `grid` ，其中：`grid[i][j] = 1` 表示陆地， `grid[i][j] = 0` 表示水域。

网格中的格子 **水平和垂直** 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。

岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        def check(grid, i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
                return True
            else:
                return False

        def dfs(grid, i, j):
            # 不能用0来标识已遍历，防止重复统计
            grid[i][j] = 2
            
            count = 0
            for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                # 从一个岛屿方格走向网格边界，周长加1
                if not check(grid, k[0], k[1]):
                    count += 1
                # 从一个岛屿方格走向水域方格，周长加1
                elif grid[k[0]][k[1]] == 0:
                    count += 1
                elif grid[k[0]][k[1]] == 1:
                    count += dfs(grid, k[0], k[1])
                # grid[k[0]][k[1]] == 2
                else:
                    continue
            return count

        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    return dfs(grid, i, j)
```

> **岛屿的周长就是岛屿方格和非岛屿方格相邻的边的数量**。注意，这里的非岛屿方格，既包括水域方格，也包括网格的边界。将这个“相邻关系”对应到 DFS 遍历中，就是：每当在 DFS 遍历中，从一个岛屿方格走向一个非岛屿方格，就将周长加 1
>
> 对于一个陆地格子的每条边，它被算作岛屿的周长当且仅当这条边为网格的边界或者相邻的另一个格子为水域

## 494 目标和

给你一个非负整数数组 `nums` 和一个整数 `target` 。

向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** ：

- 例如，`nums = [2, 1]` ，可以在 `2` 之前添加 `'+'` ，在 `1` 之前添加 `'-'` ，然后串联起来得到表达式 `"+2-1"` 。

返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

```python
# 二维dp数组
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums)
        diff = s - target
        if diff < 0 or diff % 2 != 0:
            return 0
        
        neg = diff // 2
        n = len(nums)
        dp = [[0] * (neg+1) for _ in range(n+1)]
        dp[0][0] = 1
        
        for i in range(1, n+1):
            num = nums[i-1]
            for j in range(neg+1):
                if j < num:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-num]
        
        return dp[n][neg]
    
# 一维dp数组
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums)
        diff = s - target
        if diff < 0 or diff % 2 != 0:
            return 0
        
        neg = diff // 2
        n = len(nums)
        dp = [0] * (neg+1)
        dp[0] = 1
        
        for i in range(1, n+1):
            num = nums[i-1]
            for j in reversed(range(num, neg+1)):
                dp[j] = dp[j] + dp[j-num]
        
        return dp[neg]
```

注意dp数组元素的索引需要从1开始，和背包问题的边界不完全一样

> 记数组的元素和为 sum，添加 - 号的元素之和为 neg，则其余添加 + 的元素之和为 sum−neg，得到的表达式的结果为(sum−neg)−neg=sum−2⋅neg=target，即neg=(sum−target)/2。由于数组 nums 中的元素都是非负整数，neg 也必须是非负整数，所以上式成立的前提是 sum−target 是非负偶数。若不符合该条件可直接返回 0。
>
> 若上式成立，问题转化成在数组 nums 中选取若干元素，使得这些元素之和等于 neg，计算选取元素的方案数。我们可以使用动态规划的方法求解。
>
> 定义二维数组 dp，其中 dp\[i][j] 表示在数组 nums 的前 i 个数中选取元素，使得这些元素之和等于 j 的方案数。假设数组 nums 的长度为 n，则最终答案为 dp\[n][neg]。
>
> 当没有任何元素可以选取时，元素和只能是 0，对应的方案数是 1，因此动态规划的边界条件是：dp\[0][j]=1, j=0; dp\[0][j]=0, j>0。
>
> 当 1≤i≤n 时，对于数组 nums 中的第 i 个元素 num（i 的计数从 1 开始），遍历 0≤j≤neg，计算 dp\[i][j] 的值：如果 j<num，则不能选 num，此时有 dp\[i][j]=dp\[i−1][j]；如果 j≥num，则如果不选 num，方案数是 dp\[i−1][j]，如果选 num，方案数是 dp\[i−1][j−num]，此时有 dp\[i][j]=dp\[i−1][j]+dp\[i−1][j−num]。
>

## 543 二叉树的直径

给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def depth(root):
            if not root:
                return 0
            
            # 左子树的深度，左子树的最深的路径包含的节点数
            L = depth(root.left)
            # 右子树的深度，右子树的最深的路径包含的节点数
            R = depth(root.right)
            # 包括当前节点在内的路径包含的节点数
            self.res = max(L+R+1, self.res)
            # 返回以当前节点为根的子树的深度
            return max(L, R) + 1

        self.res = 1
        depth(root)
        # 树的直径为最长路径的长度，即最长的路径包含节点数减一
        return self.res - 1
```

> 首先我们知道一条路径的长度为该路径经过的节点数减一，所以求直径（即求路径长度的最大值）等效于求路径经过节点数的最大值减一。而任意一条路径均可以被看作由某个节点为起点，从其左儿子和右儿子向下遍历的路径拼接得到。
>
> 假设我们知道对于该节点的左儿子向下遍历经过最多的节点数 L （即以左儿子为根的子树的深度） 和其右儿子向下遍历经过最多的节点数 R （即以右儿子为根的子树的深度），那么以该节点为起点的路径经过节点数的最大值即为 L+R+1。
>
> 我们记节点 node 为起点的路径经过节点数的最大值为 d_node，那么二叉树的直径就是所有节点 d_node的最大值减一。
>
> 最后的算法流程为：我们定义一个递归函数 depth(node) 计算 d_node，函数返回该节点为根的子树的深度。先递归调用左儿子和右儿子求得它们为根的子树的深度 L 和 R ，则该节点为根的子树的深度即为max(L, R)+1，而该节点的 d_node值为L+R+1，递归搜索每个节点并设一个全局变量 ans 记录 d_node的最大值，最后返回 ans-1 即为树的直径。

## 560 和为 K 的子数组

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的连续子数组的个数* 。

子数组是数组中元素的连续非空序列。

```python
# 方法一：暴力枚举，超时
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += nums[j]
                # nums元素可能为0或负数，不能找到一个就break
                if s == k:
                    res += 1
        return res
    
# 方法二：前缀和，快速计算[i,j]区间和，超时
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0
        # 前缀和数组，第一项为0
        pre_sum = [0] * (n+1)
        for i in range(n):
            pre_sum[i+1] = pre_sum[i] + nums[i]
        
        for i in range(n):
            for j in range(i, n):
                # 注意下标偏移，前缀和数组有第一项为0，索引应都加一
                s = pre_sum[j+1] - pre_sum[i]
                if s == k:
                    res += 1
        return res
```

> 做法一：暴力枚举
>
> 考虑以 i 开头和为 k 的连续子数组个数，我们需要统计符合条件的下标 j 的个数，其中 i≤j<n 且 [i..j] 这个子数组的和恰好为 k 。我们可以枚举 [i..n-1] 里所有的下标 j 来判断是否符合条件，可能有读者会认为假定我们确定了子数组的开头和结尾，还需要 O(n) 的时间复杂度遍历子数组来求和，那样复杂度就将达到 O(n3)从而无法通过所有测试用例。但是如果我们知道 [i,j] 子数组的和，就能 O(1) 推出[i,j+1] 的和，因此这部分的遍历求和是不需要的，我们在枚举下标 j 的时候已经能 O(1) 求出 [i,j] 的子数组之和。
>
> 做法二：前缀和+哈希表
>
> 看不懂

## 617 合并二叉树

给你两棵二叉树： `root1` 和 `root2` 。

想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，**不为** null 的节点将直接作为新二叉树的节点。

返回合并后的二叉树。

**注意:** 合并过程必须从两个树的根节点开始。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root1, root2):
            if not root1 and not root2:
                return
            if not root1:
                return root2
            if not root2:
                return root1
            
            root = TreeNode(root1.val + root2.val)
            root.left = dfs(root1.left, root2.left)
            root.right = dfs(root1.right, root2.right)
            return root
        
        return dfs(root1, root2)
```

## 695 岛屿的最大面积

给你一个大小为 `m x n` 的二进制矩阵 `grid` 。

**岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在 **水平或者竖直的四个方向上** 相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

岛屿的面积是岛上值为 `1` 的单元格的数目。

计算并返回 `grid` 中最大的岛屿面积。如果没有岛屿，则返回面积为 `0` 。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def check(grid, i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
                return True
            else:
                return False

        def dfs(grid, i, j):
            grid[i][j] = 0
            area = 1

            for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if check(grid, k[0], k[1]) and grid[k[0]][k[1]] == 1:
                    area += dfs(grid, k[0], k[1])
            return area

        m = len(grid)
        n = len(grid[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = dfs(grid, i, j)
                    res = max(res, area)
        return res
```

注意看矩阵中存的元素类型，200题是字符串，本题是整型

## 704 二分查找

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return -1
```

Python中要用//来整除

## 739 每日温度

给定一个整数数组 `temperatures` ，表示每天的温度，返回一个数组 `answer` ，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

```python
# 做法一：暴力，超时
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        answer = [0] * n
        for i in range(n):
            temp = temperatures[i]
            for j in range(i+1, n):
                if temperatures[j] > temp:
                    answer[i] = j - i
                    break
        return answer
    
# 做法二：单调栈
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        answer = [0] * n
        stack = []

        for i in range(n):
            temp = temperatures[i]
            while stack and temperatures[stack[-1]] < temp:
                prev_index = stack.pop()
                answer[prev_index] = i - prev_index
            stack.append(i)
        return answer
```

> 做法二：单调栈
>
> 可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。
>
> 正向遍历温度列表。对于温度列表中的每个元素 temperatures[i]，如果栈为空，则直接将 i 进栈，如果栈不为空，则比较栈顶元素 prevIndex 对应的温度 temperatures[prevIndex] 和当前温度 temperatures[i]，如果 temperatures[i] > temperatures[prevIndex]，则将 prevIndex 移除，并将 prevIndex 对应的等待天数赋为 i - prevIndex，重复上述操作直到栈为空或者栈顶元素对应的温度大于等于当前温度，然后将 i 进栈。
>
> 为什么可以在弹栈的时候更新 ans[prevIndex] 呢？因为在这种情况下，即将进栈的 i 对应的 temperatures[i] 一定是 temperatures[prevIndex] 右边第一个比它大的元素，试想如果 prevIndex 和 i 有比它大的元素，假设下标为 j，那么 prevIndex 一定会在下标 j 的那一轮被弹掉。
>
> 由于单调栈满足从栈底到栈顶元素对应的温度递减，因此每次有元素进栈时，会将温度更低的元素全部移除，并更新出栈元素对应的等待天数，这样可以确保等待天数一定是最小的。
>

## 827 最大人工岛

给你一个大小为 `n x n` 二进制矩阵 `grid` 。**最多** 只能将一格 `0` 变成 `1` 。

返回执行此操作后，`grid` 中最大的岛屿面积是多少？

**岛屿** 由一组上、下、左、右四个方向相连的 `1` 形成。

```python
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        def check(grid, i, j):
            return 0 <= i < len(grid) and 0 <= j < len(grid[0])

        def dfs(grid, i, j):
            # 用岛屿编号标记已遍历
            grid[i][j] = index
            area = 1

            for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if check(grid, k[0], k[1]) and grid[k[0]][k[1]] == 1:
                    area += dfs(grid, k[0], k[1])
            return area
        
        res = 0
        areas = {} # island index: area
        index = 2
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = dfs(grid, i, j)
                    areas.update({index: area})
                    res = max(res, area)
                    index += 1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    new_area = 1
                    # 同一岛屿的所有节点编号相同，用set防止重复统计
                    connected = set()
                    for k in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        # 不越界，且周围点为岛屿，且岛屿没被统计过
                        if check(grid, k[0], k[1]) and grid[k[0]][k[1]] != 0 and grid[k[0]][k[1]] not in connected:
                            new_area += areas[grid[k[0]][k[1]]]
                            connected.add(grid[k[0]][k[1]])
                    res = max(new_area, res)
        return res
```

> 题目要求计算“经过某些操作”之后的岛屿面积，而岛屿是不同的，所以我们可以在遍历整个矩阵的过程中，对不同的岛屿进行编号。由于0和1已经被使用了，那么岛屿的编号我们就从2开始，当遍历到新的岛屿时，岛屿编号加1。并且，在遍历过程中，将每个岛屿的面积也统计出来，并保存到Map中（key=岛屿编号；value=岛屿面积）。
>
> 为了防止dfs遍历计算岛屿面积时出现重复遍历，我们将格子值修改为当前岛屿编号。那么，终止条件如下所示：遍历格子下标已经越界；遍历的格子不为1，即不遍历海洋格子和已重新编号过的岛屿。
>
> 对于每个 grid\[i][j]=0，我们计算将它变为 1 后，新合并的岛屿的面积 z（z 的初始值为 1，对应该点的面积）：使用集合 connected 保存与 grid\[i][j] 相连的岛屿，遍历与 grid\[i][j] 相邻的四个点，如果该点的值为 1，且它所在的岛屿并不在集合中，我们将 z 加上该点所在的岛屿面积，并且将该岛屿加入集合中。所有这些新合并岛屿以及原来的岛屿的面积的最大值就是最大的岛屿面积。
>

## 912 排序数组

给你一个整数数组 `nums`，请你将该数组升序排列。

```python
# 选最左边一个数作为轴值pivot
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quicksort(nums, l, r):
            # 只有0或1个元素
            if l >= r:
                return

            pivot = nums[l]
            i, j = l, r
            while i < j:
                # 找右边第一个小于pivot的值
                while i < j and nums[j] >= pivot:
                    j -= 1
                # 找左边第一个大于pivot的值
                while i < j and nums[i] <= pivot:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l] = nums[i]
            nums[i] = pivot
            quicksort(nums, l, i-1)
            quicksort(nums, i+1, r)
        
        quicksort(nums, 0, len(nums)-1)
        return nums

# 选数组中间点或随机值作为轴值pivot，唯一不同是需要在选取后将pivot先与数组最左值交换
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quicksort(nums, l, r):
            # 只有0或1个元素
            if l >= r:
                return
            
            # 取数组中间点
            # pivot_index = (l + r) // 2
            # 随机取pivot
            pivot_index = random.randint(l, r)
            nums[l], nums[pivot_index] = nums[pivot_index], nums[l]
            
            pivot = nums[l]
            i, j = l, r
            while i < j:
                # 找右边第一个小于pivot的值
                while i < j and nums[j] >= pivot:
                    j -= 1
                # 找左边第一个大于pivot的值
                while i < j and nums[i] <= pivot:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l] = nums[i]
            nums[i] = pivot
            quicksort(nums, l, i-1)
            quicksort(nums, i+1, r)
        
        quicksort(nums, 0, len(nums)-1)
        return nums
```

快速排序

## 1143 最长公共子序列

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
```

> 假设字符串 text1 和 text2 的长度分别为 m 和 n，创建 m+1 行 n+1 列的二维数组 dp，其中 dp\[i][j] 表示 text1[0:i] 和 text2[0:j] 的最长公共子序列的长度。上述表示中，text1[0:i] 表示 text1 的长度为 i 的前缀，text2[0:j] 表示 text2 的长度为 j 的前缀。
>
> 当 i=0 时，text1[0:i] 为空，空字符串和任何字符串的最长公共子序列的长度都是 0，因此对任意 0≤j≤n，有 dp\[0][j]=0；当 j=0 时，text2[0:j] 为空，同理可得，对任意 0≤i≤m，有 dp\[i][0]=0。
>
> 当 i>0 且 j>0 时，考虑 dp\[i][j] 的计算：
>
> 当 text1[i−1]=text2[j−1] 时，将这两个相同的字符称为公共字符，考虑 text1[0:i−1] 和 text2[0:j−1] 的最长公共子序列，再增加一个字符（即公共字符）即可得到 text1[0:i] 和 text2[0:j] 的最长公共子序列，因此 dp\[i][j]=dp\[i−1][j−1]+1。
>
> 当 text1[i−1]≠text2[j−1] 时，考虑以下两项：text1[0:i−1] 和 text2[0:j] 的最长公共子序列；text1[0:i] 和 text2[0:j−1] 的最长公共子序列。要得到 text1[0:i] 和 text2[0:j] 的最长公共子序列，应取两项中的长度较大的一项，因此 dp\[i][j]=max⁡(dp\[i−1][j],dp\[i][j−1])。
>
> 最终计算得到 dp\[m][n] 即为 text1 和 text2 的最长公共子序列的长度。

## 01背包

你有一个背包，最多能容纳的体积是V。现在有n个物品，第i个物品的体积为vi，价值为wi，求这个背包最多能装多大价值的物品？

```python
# 二维dp数组：dp[n][V+1]解法
class Solution:
    def knapsack(self, V: int, n: int, v: List[int], w: List[int]) -> int:
        # 注意是n行(V+1)列，多出来的那 1 列，表示背包容量从 0 开始考虑
        dp = [[0] * (V+1) for _ in range(n)]
        # 初始化，第一列表示背包容量为0时，能装物品的最大价值也都是0，第一行表示只选取0号物品最大价值
        for j in range(V+1):
            dp[0][j] = 0 if v[0] > j else w[0]
        
        # 外层遍历物品（物品0已经初始化），内层遍历背包容量
        # 二维dp数组中遍历物品和背包的先后顺序可以交换，并且遍历背包可以正序也可以逆序reversed(range(V+1))，应该从二维dp数组当前元素dp[i][j]所依赖元素是正上方dp[i-1][j]和左上方dp[i-1][j-v[i]]来思考为什么，即只要保证在计算dp[i][j]时其正上方元素和左上方元素已经算好即可
        for i in range(1, n):
            for j in range(V+1):
                if v[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-v[i]]+w[i])
        return dp[n-1][V]

# 滚动数组：dp[2][V+1]解法，将所有dp数组行索引模2即可
class Solution:
    def knapsack(self, V: int, n: int, v: List[int], w: List[int]) -> int:
        # 2行(V+1)列
        dp = [[0] * (V+1) for _ in range(2)]
        for j in range(V+1):
            dp[0][j] = 0 if v[0] > j else w[0]
        
        for i in range(1, n):
            for j in range(V+1):
                if v[i] > j:
                    dp[i%2][j] = dp[(i-1)%2][j]
                else:
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[(i-1)%2][j-v[i]]+w[i])
        return dp[(n-1)%2][V]
    
# 一维dp数组：dp[V+1]解法
class Solution:
    def knapsack(self, V: int, n: int, v: List[int], w: List[int]) -> int:
        dp = [0] * (V + 1)
        
        # 必须先遍历物品（从物品0开始），再遍历背包，且必须倒序遍历背包，保证每个物品只能使用一次
        for i in range(n):
            # 如果和二维dp数组的遍历范围保持一致，即reversed(range(V+1))，就要加上if v[i]>j: dp[j]=dp[j]，和二维dp数组的if-else判断写法一样，但其实因为j<v[i]时dp[j]不需要更新，所以可以直接遍历范围reversed(range(v[i], V+1))
            for j in reversed(range(v[i], V+1)):
                dp[j] = max(dp[j], dp[j-v[i]]+w[i])
        return dp[V]
```

> dp\[i][j]表示从 0 到 i 个物品中选择不超过 j 重量的物品的最大价值。
>
> 滚动数组：根据二维dp数组「转移方程」，我们知道计算第 i 行格子只需要第 i-1 行中的某些值，也就是计算「某一行」的时候只需要依赖「前一行」，因此可以用一个只有两行的数组来存储中间结果，根据当前计算的行号是偶数还是奇数来交替使用第 0 行和第 1 行，这样的空间优化方法称为「滚动数组」。只需要将代表行的维度修改成 2，并将所有使用dp数组行维度的地方从 i 改成 i % 2 或者 i & 1即可（更建议使用 i & 1，& 运算在不同 CPU 架构的机器上要比 % 运算稳定）。
>
> 一维dp数组：dp[j]表示容量为 j 的背包所能装的物品的最大价值。dp全初始化为0，为了保证状态递推时不影响max函数的执行。求解第 i 行格子的值时，不仅是只依赖第 i-1 行，还明确只依赖第 i-1 行的第 j 个格子和第 j-v[i] 个格子，即只依赖于「上一个格子的位置」以及「上一个格子的左边位置」。因此，只要我们将求解第 i 行格子的顺序「从 0 到 V 」改为「从 V 到 0 」，就可以将原本 2 行的滚动数组压缩到一行（转换为一维数组），可以确保我们在更新某个状态时，所需要用到的状态值不会被覆盖。

## 完全背包

你有一个背包，最多能容纳的体积是V。现在有n种物品，每种物品有任意多个，第i种物品的体积为vi,价值为wi。（1）求这个背包至多能装多大价值的物品？（2）若背包恰好装满，求至多能装多大价值的物品？

```python
# 二维dp数组：dp[n][V+1]
class Solution:
    def knapsack(self, V: int, n: int, v: List[int], w: List[int]) -> int:
        dp = [[0] * (V+1) for _ in range(n)]
        
        # 只有一件物品时，在容量允许的情况下，能选多少件就选多少件
        for j in range(V+1):
            # 也可以直接写成dp[0][j]=(j//v[0])*w[0]，因为//是向下取整
            dp[0][j] = 0 if v[0] > j else (j//v[0])*w[0]
        
        # 物品和背包先后顺序可以替换，背包也可以逆序遍历
        for i in range(1, n):
            for j in range(V+1):
                if v[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    # 计算物品i的最大可选数目k
                    k = 1
                    while v[i]*(k+1) <= j:
                        k += 1
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-k*v[i]]+k*w[i])
        return dp[n-1][V]

# 滚动数组：dp[2][V+1]
class Solution:
    def knapsack(self, V: int, n: int, v: List[int], w: List[int]) -> int:
        dp = [[0] * (V+1) for _ in range(2)]
        
        for j in range(V+1):
            dp[0][j] = 0 if v[0] > j else (j//v[0])*w[0]
        
        for i in range(1, n):
            for j in range(V+1):
                if v[i] > j:
                    dp[i%2][j] = dp[(i-1)%2][j]
                else:
                    k = 1
                    while v[i]*(k+1) <= j:
                        k += 1
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[(i-1)%2][j-k*v[i]]+k*w[i])
        return dp[(n-1)%2][V]
    
# 一维dp数组：dp[V+1]
class Solution:
    def knapsack(self, v: int, n: int, v: List[int], w: List[int]) -> int:
        dp = [0] * (V+1)
        dp2 = [-inf] * (V+1)
        dp2[0] = 0
        # 物品和背包先后顺序可以交换，交换后j的范围就是0到V+1，需要加上if v[i]>j: dp[j]=dp[j]
        for i in range(n):
            for j in range(v[i], V+1):
                dp[j] = max(dp[j], dp[j-v[i]]+w[i])
                dp2[j] = max(dp2[j], dp2[j-v[i]]+w[i])
        return [dp[V], int(dp2[V]) if dp[2] > 0 else 0]
```

> 形式上，我们只需要将 01 背包问题的「一维dp数组」解法中的「容量维度」遍历方向从「从大到小 改为 从小到大」就可以解决完全背包问题。
>
> 求最优解的背包问题中，有的题目要求恰好装满背包时的最优解，有的题目则要求不超过背包容量时的最优解。一种区别这两种问法的实现方法是在状态初始化的时候有所不同。初始化的 dp 数组事实上就是在背包中没有放入任何物品时的合法状态：
>
> 如果要求恰好装满背包，那么在初始化时 dp\[i][0]=0，其它 dp\[i][1,2,...,∗] 均设为 −∞。这是因为此时只有容量为 0 的背包可能被价值为 0 的 nothing “恰好装满”，而其它容量的背包均没有合法的解，属于未定义的状态。如果只是要求不超过背包容量而使得背包中的物品价值尽量大，初始化时应将 dp\[∗][∗] 全部设为 0。这是因为对应于任何一个背包，都有一个合法解为 “什么都不装”，价值为 0。

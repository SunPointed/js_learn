### 1
```
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```
##### mine
```
class Solution {
    fun generateParenthesis(n: Int): List<String> {
        if (n <= 0) return emptyList()

        val res = mutableSetOf<String>()

        generateParenthesis(n * 2 - 1, "(", res)

        return res.toList()
    }
    
    fun generateParenthesis(n: Int, cur: String, set: MutableSet<String>) {
        if (n > 0) {
            generateParenthesis(n - 1, "$cur(", set)
            generateParenthesis(n - 1, "$cur)", set)
        } else {
            if (isValid(cur)) {
                set.add(cur)
            }
        }
    }

    fun isValid(value: String): Boolean {
        var balance = 0

        var index = 0
        while (index < value.length) {
            val cur = value[index]
            if (cur == ')') {
                balance--
            } else {
                balance++
            }
            if(balance < 0) return false
            
            index++
        }
        return balance == 0
    }
}
```
##### like
```
class Solution {
    public List<String> generateParenthesis(int n) {
        if(n==1) return new ArrayList<String>(Arrays.asList("()"));
        Set<String> set=new HashSet<String>();
        for(String str:generateParenthesis(n-1)){
            for(int i=0;i<str.length();i++){
                set.add(str.substring(0,i+1)+"()"+str.substring(i+1,str.length()));               
            }            
        }
        List<String> list = new ArrayList<String>(set);
        return list;
        
    }
}
```
### 2
```
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

Each number in candidates may only be used once in the combination.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.

Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```
#### mine 1
```
class Solution {
    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
        val res = mutableSetOf<List<Int>>()
        sub(candidates, target, null, res)
        return res.toList()
    }
    
    fun sub(candidates: IntArray, target: Int, preIndexList:MutableList<Int>?, resList: MutableSet<List<Int>>) {
        candidates.forEachIndexed { index, data ->
            if(preIndexList == null || !preIndexList.contains(index)) {
                when {
                    data > target -> {

                    }
                    data < target -> {
                        val preList =  mutableListOf<Int>().apply {
                            preIndexList?.also {
                                addAll(it)
                            }
                            add(index)
                        }
                        sub(candidates, target - data, preList, resList)
                    }
                    else -> {
                        resList.add(mutableListOf<Int>().apply {
                            preIndexList?.forEach {
                                this.add(candidates[it])
                            }
                            this.add(data)
                            sort()
                        })
                    }
                }
            }
        }
    }
}
```
##### mine 2
```
class Solution {
    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        sub(candidates.apply { sort() }, 0, target, null, res)
        return res
    }
    
    fun sub(
        candidates: IntArray,
        startIndex: Int,
        target: Int,
        datas: MutableList<Int>?,
        resList: MutableList<List<Int>>
    ) {
        for (i in startIndex until candidates.size) {
            if (i > startIndex && candidates[i] == candidates[i - 1]) continue

            val data = candidates[i]
            if (data > target) continue
            else if (data < target) {
                sub(candidates, i + 1, target - data, mutableListOf<Int>().apply {
                    datas?.also {
                        addAll(it)
                    }
                    add(data)
                }, resList)
            } else {
                resList.add(mutableListOf<Int>().apply {
                    datas?.also {
                        addAll(it)
                    }
                    add(data)
                })
            }
        }
    }
}
```
##### understand
```
class Solution {
    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        subSum(candidates.apply { sort() }, 0, target, mutableListOf(), res)
        return res
    }
    
    private fun subSum(
        candidates: IntArray,
        curIndex: Int,
        target: Int,
        temp: MutableList<Int>,
        res: MutableList<List<Int>>
    ) {
        if (target == 0) {
            res.add(mutableListOf<Int>().apply { addAll(temp) })
            return
        }
        if (target < 0) {
            return
        }
        for (i in curIndex until candidates.size) {
            if(i > curIndex && candidates[i] == candidates[i - 1]) {
                continue
            }
            temp.add(candidates[i])
            subSum(candidates, i + 1, target - candidates[i], temp, res)
            temp.removeAt(temp.size - 1)
        }
    }
}
```
### 3
```
Multiply Strings

Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Input: num1 = "2", num2 = "3"
Output: "6"

Input: num1 = "123", num2 = "456"
Output: "56088"

Note:
1.The length of both num1 and num2 is < 110.
2.Both num1 and num2 contain only digits 0-9.
3.Both num1 and num2 do not contain any leading zero, except the number 0 itself.
4.You must not use any built-in BigInteger library or convert the inputs to integer directly.
```
##### mine
```
class Solution {
    fun multiply(num1: String, num2: String): String {
        if (num1 == "0" || num2 == "0") return "0"

        val m = Array<Array<Int?>>(num1.length) {
            arrayOfNulls(num2.length)
        }
        num1.forEachIndexed { x1, c1 ->
            num2.forEachIndexed { x2, c2 ->
                m[x1][x2] = c1.int * c2.int
            }
        }

        var res = ""
        var next = 0
        val maxStep = num1.length - 1 + num2.length - 1
        var curStep = maxStep
        while (curStep >= 0) {
            var cur = 0

            for (i in num1.indices) {
                for (j in num2.indices) {
                    if (i + j == curStep) {
                        m[i][j]?.also {
                            cur += it
                        }
                    }
                }
            }

            cur += next
            res = "${cur % 10}$res"
            next = cur / 10
            curStep--
        }

        while (next != 0) {
            res = "${next % 10}$res"
            next /= 10
        }

        return res
    }

    private val Char.int: Int
        get() = when (this) {
            '0' -> 0
            '1' -> 1
            '2' -> 2
            '3' -> 3
            '4' -> 4
            '5' -> 5
            '6' -> 6
            '7' -> 7
            '8' -> 8
            else -> 9
        }
}
```
##### like
```
class Solution {
    fun multiply(num1: String, num2: String): String {
        val l1 = num1.length - 1
        val l2 = num2.length - 1
        val a = IntArray(l1 + l2 + 2) {
            0
        }
        for(i in (0..l1).reversed()){
            for(j in (0..l2).reversed()) {
                val t = (num1[i] - '0') * (num2[j] - '0')
                val temp = a[i + j + 1] + t
                a[i + j + 1] = temp % 10
                a[i + j] += temp / 10
            }
        }

        var value = ""
        a.forEachIndexed { index, i ->
            if(value != "" || i != 0){
                value += i
            }
        }
        return if(value == "") "0" else value
    }
}
```
### 4
```
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Input: "()"
Output: true

Input: "(]"
Output: false

Input: "([)]"
Output: false

Input: "{[]}"
Output: true
```
##### mine
```
class Solution {
    fun isValid(s: String): Boolean {
        if (s.isEmpty()) return true
        // ArrayDeque代替Stack，注意是从最前端入栈出栈
        val stack = ArrayDeque<Char>()
        s.forEach {
            if (stack.size > 0) {
                if (needPop(it, stack.first)) {
                    stack.pop()
                } else {
                    stack.push(it)
                }
            } else {
                stack.push(it)
            }
        }
        return stack.size == 0
    }

    fun needPop(x1: Char, x2: Char): Boolean {
        return (x1 == ')' && x2 == '(') || (x1 == ']' && x2 == '[' || (x1 == '}' && x2 == '{'))
    }
}
```
### 5
```
Implement pow(x, n), which calculates x raised to the power n (xn).

Input: 2.00000, 10
Output: 1024.00000

Input: 2.10000, 3
Output: 9.26100

Input: 2.00000, -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25

Note:
    -100.0 < x < 100.0
    n is a 32-bit signed integer, within the range [−231, 231 − 1]
```
##### like
```
class Solution {
    fun myPow(x: Double, n: Int): Double {
        if (n == 1) return x
        if (n == -1) return 1.0 / x
        if (n == 0) return 1.0
        val half = myPow(x, n / 2)
        return half * half * myPow(x, n % 2)
    }
}
```
### 6
```
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
Empty cells are indicated by the character '.'.

Note:
The given board contain only digits 1-9 and the character '.'.
You may assume that the given Sudoku puzzle will have a single unique solution.
The given board size is always 9x9.
```
##### mine
```
import kotlin.experimental.or
class Solution {
    fun solveSudoku(board: Array<CharArray>): Unit {
        solve(board, 0, 0)
    }
    
    fun solve(board: Array<CharArray>, x: Int, y: Int): Boolean {
        if (isSolve(board)) return true

        val ny = (y + 1) % 9
        val nx = if(ny == 0) (x + 1) % 9 else x
        if (board[x][y] == '.') {
            var solve = false
            for (index in 0..8) {
                board[x][y] = '1' + index
                if (isValid(board, x, y, board[x][y])) {
                    if (solve(board, nx, ny)) return true
                }
            }
            board[x][y] = '.'
            return false
        } else {
            return solve(board, nx, ny)
        }
    }
    
    fun isValid(board: Array<CharArray>, x: Int, y: Int, cur: Char): Boolean {
        var res = true

        res = board[x].filterIndexed { index, _ ->
            index != y
        }.none {
            it == cur
        }

        if (!res) return false

        board.forEachIndexed { index, chars ->
            if (index != x && chars[y] == cur) {
                return false
            }
        }

        val sx = when {
            x < 3 -> 0
            x < 6 -> 3
            else -> 6
        }

        val sy = when {
            y < 3 -> 0
            y < 6 -> 3
            else -> 6
        }

        for (i in sx until (sx + 3)) {
            val b = board[i]
            for (j in sy until (sy + 3)) {
                val c = b[j]
                if (!(i == x && j == y) && c == cur) {
                    return false
                }
            }
        }

        return res
    }

    fun isSolve(board: Array<CharArray>): Boolean {
        return board.all { chars ->
            val noDot = chars.none { c ->
                c == '.'
            }
            noDot
        }
    }
}
```
### 7
```
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```
##### mine 1
```
class Solution {
    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        var resNode: ListNode? = null
        var tempNode: ListNode? = null

        var curMin = findMin(lists)
        while (curMin != null) {
            if (resNode == null) {
                resNode = curMin
                tempNode = curMin
            } else {
                tempNode?.next = curMin
                tempNode = curMin
            }
            curMin = findMin(lists)
        }

        return resNode
    }

    fun findMin(lists: Array<ListNode?>): ListNode? {
        var min: ListNode? = null
        var index = -1
        lists.forEachIndexed { i, listNode ->
            if (listNode != null) {
                if (min == null) {
                    min = listNode
                    index = i
                } else {
                    if (min!!.`val` > listNode.`val`) {
                        min = listNode
                        index = i
                    }
                }
            }
        }

        if(index != -1 && min != null) {
            lists[index] = min?.next
        }
        return min
    }
    
    class ListNode(var `val`: Int) {
        var next: ListNode? = null
    }
}
```
##### mine 2
```
class Solution {
    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        if (lists.isEmpty()) return null
        if (lists.size == 1) return lists[0]

        var len = lists.size
        while (len != 1) {
            for (i in 0 until (len / 2)) {
                lists[i] = mergeTwo(lists[i * 2], lists[i * 2 + 1])
            }
            if(len % 2 == 1){
                lists[len / 2] = lists[len - 1]
                len = (len + 1) / 2
            } else {
                len /= 2
            }
        }
        return lists[0]
    }

    fun mergeTwo(node1: ListNode?, node2: ListNode?): ListNode? {
        if (node1 == null) return node2
        if (node2 == null) return node1
        val start = ListNode(0)
        var cur = start
        var n1 = node1
        var n2 = node2
        while (n1 != null || n2 != null) {
            if(n1 != null && n2 !=null) {
                if (n1.`val` <= n2.`val`) {
                    cur.next = n1
                    cur = n1
                    n1 = n1.next
                } else {
                    cur.next = n2
                    cur = n2
                    n2 = n2.next
                }
            } else if(n1 != null) {
                cur.next = n1
                n1 = null
            } else if(n2 != null) {
                cur.next = n2
                n2 = null
            }
        }
        return start.next
    }
}
```
### 8
```
Given an unsorted integer array, find the smallest missing positive integer.

Input: [1,2,0]
Output: 3

Input: [3,4,-1,1]
Output: 2

Input: [7,8,9,11,12]
Output: 1

Your algorithm should run in O(n) time and uses constant extra space.
```
##### mine 1 too slow
```
class Solution {
    class Node(
        var value: Int,
        var next: Node? = null
    )
    
    fun firstMissingPositive(nums: IntArray): Int {
        val head = Node(-1)
        var count = 0
        nums.forEach {
            if (it > 0) {
                val node = Node(it)
                var cur = head.next
                if (cur == null) {
                    head.next = node
                } else {
                    var find = false
                    var pre:Node? = head
                    while (cur != null) {
                        if (cur!!.value < it) {
                            pre = cur
                            cur = cur!!.next
                        } else if(cur!!.value == it){
                            find = true
                            break
                        } else {
                            find = true

                            pre?.next = node
                            node.next = cur

                            break
                        }
                    }

                    if (!find) {
                        pre?.next = node
                    }
                }
                count++
            }
        }

        return if(head.next == null || head.next!!.value > 1) {
            1
        } else {
            var value = 1
            var cur = head.next
            while (cur != null && cur!!.value == value){
                cur = cur!!.next
                value++
            }

            value
        }
    }
}
```
##### mine 2 even slow
```
class Solution {
    fun firstMissingPositive(nums: IntArray): Int {
        nums.sort()
        var value = 1
        nums.forEach {
            if (it > 0 && it == value) {
                value++
            }
        }
        return value   
    }
}
```
##### mine 3 acceptable
```
class Solution {
    fun firstMissingPositive(nums: IntArray): Int {
        var count = 0
        val set = mutableSetOf<Int>()
        nums.forEach {
            if (it > 0) {
                set.add(it)
                count++
            }
        }

        for (res in 1..count) {
            if (!set.contains(res)) {
                return res
            }
        }

        return 1 + count 
    }
}
```
##### mine 4 best
```
class Solution {
    fun firstMissingPositive(nums: IntArray): Int {
        var i = 0
        while (i < nums.size) {
            if(nums[i] == i + 1 || nums[i] < 1 || nums[i] > nums.size || nums[nums[i] - 1] == nums[i])
                i++
            else {
                val temp = nums[i]
                nums[i] = nums[temp - 1]
                nums[temp - 1] = temp
            }
        }

        i = 0
        var res = 1
        while (i < nums.size){
            if(nums[i] == res){
                i++
                res++
            } else {
                break
            }
        }

        return res
    }
}
```
### 9
```
Given an array of strings, group anagrams together.

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

All inputs will be in lowercase.
The order of your output does not matter.
```
##### mine 1 acceptable
```
class Solution {
    fun groupAnagrams(strs: Array<String>): List<List<String>> {
        val map = mutableMapOf<String, MutableList<String>>()
        strs.forEach {
            val key = it.toList().sorted().joinToString()
            if(map[key] == null) {
                map[key] = mutableListOf()
            }

            map[key]?.add(it)
        }

        return map.map {
            it.value
        }
    }
}
```
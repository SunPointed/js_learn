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
### 10
```
Substring with Concatenation of All Words

You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.

Example 1:

Input:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
Output: [0,9]

Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.

Example 2:

Input:
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
Output: []
```
##### 1
```
class Solution {
    fun findSubstring(s: String, words: Array<String>): List<Int> {
        val res = mutableListOf<Int>()
        val wordNum = words.size
        if (wordNum == 0) {
            return res
        }
        val wordLen = words[0].length
        //HashMap1 存所有单词
        val allWords = mutableMapOf<String, Int>()
        for (w in words) {
            val value = allWords.getOrDefault(w, 0)
            allWords[w] = value + 1
        }
        //遍历所有子串
        for (i in 0..(s.length - wordNum * wordLen)) {
            //HashMap2 存当前扫描的字符串含有的单词
            val hasWords = mutableMapOf<String, Int>()
            var num = 0
            //判断该子串是否符合
            while (num < wordNum) {
                val word = s.substring(i + num * wordLen, i + (num + 1) * wordLen)
                //判断该单词在 HashMap1 中
                if (allWords.containsKey(word)) {
                    val value = hasWords.getOrDefault(word, 0)
                    hasWords[word] = value + 1
                    //判断当前单词的 value 和 HashMap1 中该单词的 value
                    if (hasWords[word]!! > allWords[word]!!) {
                        break
                    }
                } else {
                    break
                }
                num++
            }
            //判断是不是所有的单词都符合条件
            if (num == wordNum) {
                res.add(i)
            }
        }
        return res
    }
}
```
### 11
```
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:
The solution set must not contain duplicate triplets.

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```
##### mine
```
class Solution {
    fun threeSum(nums: IntArray): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        nums.sort()
        for ((index, num) in nums.withIndex()) {
            if (index > 0 && nums[index - 1] == num) continue

            twoSum1(index + 1, nums, -num).forEach {
                res.add(it.apply {
                    add(num)
                })
            }

        }
        return res
    }
    
    fun twoSum1(startIndex: Int, nums: IntArray, target: Int): List<MutableList<Int>> {
        val res = mutableListOf<MutableList<Int>>()
        var i = startIndex
        var j = nums.size - 1
        while (i < j) {
            val t = nums[i] + nums[j]
            if (t == target) {
                res.add(mutableListOf(nums[i], nums[j]))
                i++
                j--
                while (i < j && nums[i - 1] == nums[i]) i++
                while (i < j && nums[j + 1] == nums[j]) j--
            } else if (t < target) {
                i++
            } else {
                j--
            }
        }
        return res
    }
}
```
### 12
```
 Construct Binary Search Tree from Preorder Traversal
 根据前序遍历结果回复二叉搜索树
 https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/

 Return the root node of a binary search tree that matches the given preorder traversal.

(Recall that a binary search tree is a binary tree where for every node, any descendant of node.left has a value < node.val, and any descendant of node.right has a value > node.val.  Also recall that a preorder traversal displays the value of the node first, then traverses node.left, then traverses node.right.)

It's guaranteed that for the given test cases there is always possible to find a binary search tree with the given requirements.

Input: [8,5,1,7,10,12]
Output: [8,5,10,1,7,null,12]
```
##### mine
```
/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
class Solution {
    fun bstFromPreorder(preorder: IntArray): TreeNode? {
        if (preorder.isEmpty()) return null

        val root = TreeNode(preorder[0])
        var curNode = root
        val stack = ArrayDeque<TreeNode>()
        var index = 1
        while (index != preorder.size) {
            if (preorder[index] < curNode.`val`) {
                stack.push(curNode)
                curNode.left = TreeNode(preorder[index])
                curNode = curNode.left!!
            } else {
                var tempNode = stack.peek()
                while (tempNode != null && tempNode.`val` < preorder[index]) {
                    curNode = stack.pop()
                    tempNode = stack.peek()
                }
                curNode.right = TreeNode(preorder[index])
                curNode = curNode.right!!
            }
            index++
        }
        return root
    }
}
```
##### recursion
```
/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
class Solution {
    var i = 0
    fun bstFromPreorder(preorder: IntArray): TreeNode? {
        return bstFromPreorder(preorder, Int.MAX_VALUE)
    }

    fun bstFromPreorder(preorder: IntArray, bound: Int): TreeNode? {
        if (i == preorder.size || preorder[i] > bound) return null
        val root = TreeNode(preorder[i++])
        root.left = bstFromPreorder(preorder, root.`val`)
        root.right = bstFromPreorder(preorder, bound)
        return root
    }
}
```
### 13
```
Reorder Routes to Make All Paths Lead to the City Zero

https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/

There are n cities numbered from 0 to n-1 and n-1 roads such that there is only one way to travel between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction because they are too narrow.

Roads are represented by connections where connections[i] = [a, b] represents a road from city a to b.

This year, there will be a big event in the capital (city 0), and many people want to travel to this city.

Your task consists of reorienting some roads such that each city can visit the city 0. Return the minimum number of edges changed.

It's guaranteed that each city can reach the city 0 after reorder.
```
##### 1
```
class Solution {
    fun minReorder(n: Int, connections: Array<IntArray>): Int {
        val right = mutableMapOf<Int, MutableSet<Int>>()
        val error = mutableMapOf<Int, MutableSet<Int>>()

        for (i in 0..n) {
            right[i] = mutableSetOf()
            error[i] = mutableSetOf()
        }

        for (road in connections) {
            right[road[1]]!!.add(road[0])
            error[road[0]]!!.add(road[1])
        }

        return stepN(0, right, error)
    }

    fun stepN(
        city: Int,
        right: MutableMap<Int, MutableSet<Int>>,
        error: MutableMap<Int, MutableSet<Int>>
    ): Int {
        var res = 0
        for (tr in right[city]!!) {
            error[tr]!!.remove(city)
            res += stepN(tr, right, error)
        }

        for (te in error[city]!!) {
            right[te]!!.remove(city)
            res++
            res += stepN(te, right, error)
        }

        return res
    }
}
```
### 14
```
Scramble String

Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.

Below is one possible representation of s1 = "great":
   great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
To scramble the string, we may choose any non-leaf node and swap its two children.

For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".
   rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
We say that "rgeat" is a scrambled string of "great".
Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".
    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
We say that "rgtae" is a scrambled string of "great".

Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.
```
##### 1
```
class Solution {
    fun isScramble(s1: String, s2: String): Boolean {
        if (s1.length != s2.length) return false
        if (s1.length == 1) return s1 == s2
        val a = s1.toCharArray().apply {
            sort()
        }
        val b = s2.toCharArray().apply {
            sort()
        }
        if(!a.contentEquals(b)) return false
        for (pos in s1.indices) {
            if ((isScramble(s1.substring(0, pos), s2.substring(0, pos)) && isScramble(
                    s1.substring(
                        pos
                    ), s2.substring(pos)
                ))
            ) {
                return true
            }
            if ((isScramble(
                    s1.substring(0, pos),
                    s2.substring(s1.length - pos)
                ) && isScramble(s1.substring(pos), s2.substring(0, s1.length - pos)))
            ) {
                return true
            }
        }
        return false
    }
}
```
### 15
```
Domino and Tromino Tiling

We have two types of tiles: a 2x1 domino shape, and an "L" tromino shape. These shapes may be rotated.

XX  <- domino

XX  <- "L" tromino
X

Given N, how many ways are there to tile a 2 x N board? Return your answer modulo 10^9 + 7.

(In a tiling, every square must be covered by a tile. Two tilings are different if and only if there are two 4-directionally adjacent cells on the board such that exactly one of the tilings has both squares occupied by a tile.)

Example:
Input: 3
Output: 5
Explanation: 
The five different ways are listed below, different letters indicates different tiles:
XYZ XXZ XYY XXY XYY
XYZ YYZ XZZ XYY XXY

N  will be in range [1, 1000].
```
##### 1
```
 /**
     * 0(1) 1(1) 2(2)2   3(5)3   3   3   3   4    4    4    4    4    4    4    4    4    4    4
     * return 0          num(n-1)+num(n-2)+2=2+1+2=5     num(n-1)+num(n-2)+num(n-3)*2+2=5+2+1*2+2=11
     *      Y    YX XX   YXZ YXX XXZ XXY XYY YXZA YXXA XXZA XXYA XYYA YXZZ XXZZ YXXY YXYY XZZY XXYY
     *      Y    YX YY   YXZ YZZ YYZ XYY XXY YXZA YZZA YYZA XYYA XXYA YXAA YYAA YXYY YXXY XXYY XZZY
     * 5     5     5     5     5     5     5          5
     * 0 ->5 0->5  1->5  1->5  2->5  2->5  3->5       4->5
     * XXZZY XZZYY YXAAZ YXXZZ YXAAZ XXZAA num(3)+ AA num(4) + M
     * XZZYY XXZZY YXXZZ YXAAZ YXAZZ YYZZA         MM          M
     * n(0) = 1
     * n(1) = 1
     * n(2) = 2
     * n > 3 -> num(n) = num(n-1) + num(n-2) + 2(num(n-3)+ ... + num(0))
     *                 = num(n-1) + [num(n-2) + num(n-3) + 2(num(n-4) + ... + num(0))] + num(n-3)
     * n > 4 -> num(n-1) = num(n-2) + num(n-3) + + 2(num(n-4) + ... + num(0))
     * n > 4 -> num(n) = num(n-1) + num(n-1) + num(n-3)
     *                 = num(n-1) * 2 + num(n-3)
     */
    fun numTilings(N: Int): Int {
        return (when (N) {
            0 -> 0
            1 -> 1
            2 -> 2
            else -> {
                val f = IntArray(N + 1)
                f[1] = 1
                f[2] = 2
                f[3] = 5
                for (i in 4..N) {
                    f[i] = 2 * f[i - 1] % 1000000007 + f[i - 2] % 1000000007
                    f[i] %= 1000000007
                }
                f[N]
            }
        })
    }
```
### 16
```
X of a Kind in a Deck of Cards

In a deck of cards, each card has an integer written on it.

Return true if and only if you can choose X >= 2 such that it is possible to split the entire deck into 1 or more groups of cards, where:

Each group has exactly X cards.
All the cards in each group have the same integer.

Input: deck = [1,2,3,4,4,3,2,1]
Output: true
Explanation: Possible partition [1,1],[2,2],[3,3],[4,4].

Input: deck = [1,1,1,2,2,2,3,3]
Output: false´
Explanation: No possible partition.

Input: deck = [1]
Output: false
Explanation: No possible partition.

Input: deck = [1,1]
Output: true
Explanation: Possible partition [1,1].

Input: deck = [1,1,2,2,2,2]
Output: true
Explanation: Possible partition [1,1],[2,2],[2,2].

Constraints:
1 <= deck.length <= 10^4
0 <= deck[i] < 10^4
```
##### mine
```
class Solution {
    fun hasGroupsSizeX(deck: IntArray): Boolean {
        val map = mutableMapOf<Int, Int>()
        for (number in deck) {
            if (map[number] == null) {
                map[number] = 1
            } else {
                map[number] = map[number]!! + 1
            }
        }
        var min = Int.MAX_VALUE
        map.values.forEach {
            if (min > it) {
                min = it
            }
        }
        var res = 0
        while (min > 1) {
            res = 0
            for (value in map.values) {
                if (value % min == 0) {
                    res++
                } else {
                    break
                }
            }
            if(res == map.values.size) return true
            min--
        }
        return res == map.values.size
    }
}
```
##### like
```
class Solution {
    fun hasGroupsSizeX(deck: IntArray): Boolean {
        val map = mutableMapOf<Int, Int>()
        for (number in deck) {
            map[number] = map.getOrDefault(number, 0) + 1
        }
        var res = 0
        for (d in map.values) {
            res = gcd(d, res)
        }
        return res > 1
    }

    // 最大公约数
    fun gcd(a: Int, b: Int): Int {
        return if (b > 0) gcd(b, a % b) else a
    }
}
```
### 17
```
Subarray Sums Divisible by K

Given an array A of integers, return the number of (contiguous, non-empty) subarrays that have a sum divisible by K.

Input: A = [4,5,0,-2,-3,1], K = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by K = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]

Note:
1 <= A.length <= 30000
-10000 <= A[i] <= 10000
2 <= K <= 10000
```
##### like
```
class Solution {
    fun subarraysDivByK(A: IntArray, K: Int): Int {
        val map = IntArray(K)

        var sum = 0
        for (a in A) {
            sum += a
            var group = sum % K
            if(group < 0) group += K
            map[group]++
        }

        var res = 0
        for(x in map) {
            if(x > 1){
                // Cn2
                res += (x * (x-1)) / 2
            }
        }

        return map[0] + res
    }
}
```
### 18
```
Design Twitter

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:

postTweet(userId, tweetId): Compose a new tweet.
getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
follow(followerId, followeeId): Follower follows a followee.
unfollow(followerId, followeeId): Follower unfollows a followee.
```
##### mine
```
class Twitter() {

    /** Initialize your data structure here. */
    private val followMap = mutableMapOf<Int, MutableList<Int>>()
    private val userList = mutableListOf<Int>()
    private val tweetMap = mutableMapOf<Int, MutableList<Tweet>>()
    private var time = 0L

    private fun initUser(userId: Int) {
        if (!userList.contains(userId)) {
            userList.add(userId)
        }
    }

    private fun initTweet(userId: Int) {
        if (!tweetMap.containsKey(userId)) {
            tweetMap[userId] = mutableListOf()
        }
    }

    private fun initFollow(userId: Int) {
        if (!followMap.containsKey(userId)) {
            followMap[userId] = mutableListOf()
        }
    }

    /** Compose a new tweet. */
    fun postTweet(userId: Int, tweetId: Int) {
        initUser(userId)
        initTweet(userId)
        tweetMap[userId]?.add(Tweet(tweetId, time++))
    }

    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    fun getNewsFeed(userId: Int): List<Int> {
        val res = mutableListOf<Tweet>()
        var watchOwn = false
        followMap[userId]?.forEach { followeeId ->
            if (followeeId == userId) {
                watchOwn = true
            }
            tweetMap[followeeId]?.also { tweets ->
                res.addAll(tweets)
            }
        }

        if (!watchOwn) {
            tweetMap[userId]?.also { tweets ->
                res.addAll(tweets)
            }
        }

        res.run {
            sortBy {
                it.time
            }
            reverse()
        }

        return if (res.size < 10) {
            res.map { it.tweetId }
        } else {
            res.subList(0, 10).map {
                it.tweetId
            }
        }
    }

    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    fun follow(followerId: Int, followeeId: Int) {
        initUser(followerId)
        initUser(followeeId)
        initFollow(followerId)

        if (followMap[followerId]?.contains(followeeId) == false) {
            followMap[followerId]?.add(followeeId)
        }
    }

    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    fun unfollow(followerId: Int, followeeId: Int) {
        initUser(followerId)
        initUser(followeeId)

        if (followMap[followerId]?.contains(followeeId) == true) {
            followMap[followerId]?.remove(followeeId)
        }

    }

    class Tweet(val tweetId: Int, val time: Long)

}

/**
 * Your Twitter object will be instantiated and called as such:
 * var obj = Twitter()
 * obj.postTweet(userId,tweetId)
 * var param_2 = obj.getNewsFeed(userId)
 * obj.follow(followerId,followeeId)
 * obj.unfollow(followerId,followeeId)
 */
```
##### like
```
class Twitter() {
    Map<Integer, List<Tweet>> tweets = new HashMap<>(); // userid -> user's tweets
    Map<Integer, Set<Integer>> followees = new HashMap<>(); // userid -> user's followees

    /** Initialize your data structure here. */
    public Twitter() {

    }

    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
        if (!tweets.containsKey(userId)) tweets.put(userId, new LinkedList<>());
        tweets.get(userId).add(0, new Tweet(tweetId));
    }

    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
        Queue<Feed> q = new PriorityQueue<>(Comparator.comparing(f -> -f.curr.order)); // descending

        if (!tweets.getOrDefault(userId, Collections.emptyList()).isEmpty()) {
            q.offer(new Feed(tweets.get(userId)));
        }

        for (Integer followee : followees.getOrDefault(userId, Collections.emptySet())) {
            if (!tweets.getOrDefault(followee, Collections.emptyList()).isEmpty()){
                q.offer(new Feed(tweets.get(followee)));
            }
        }

        List<Integer> feeds = new ArrayList<>();
        for (int i = 0; i < 10 && !q.isEmpty(); i++) {
            Feed feed = q.poll();
            feeds.add(feed.curr.id);

            if (feed.advance()) {
                q.offer(feed);
            }
        }

        return feeds;
    }

    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
        if (followerId == followeeId) return;
        if (!followees.containsKey(followerId)) followees.put(followerId, new HashSet<>());
        followees.get(followerId).add(followeeId);
    }

    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
        if (!followees.containsKey(followerId)) return;
        followees.get(followerId).remove(followeeId);
    }

    int globalOrder = 0;

    class Tweet {
        int id;
        int order;

        Tweet(int id) {
            this.id = id;
            this.order = globalOrder++;
        }
    }

    class Feed {
        Iterator<Tweet> iterator;
        Tweet curr;

        Feed(List<Tweet> tweets) {
            // tweets cannot be empty
            iterator = tweets.iterator();
            curr = iterator.next();
        }

        boolean advance() {
            if (!iterator.hasNext()) return false;
            this.curr = iterator.next();
            return true;
        }
    }
}
```
### 19
```
Longest Harmonious Subsequence

We define a harmounious array as an array where the difference between its maximum value and its minimum value is exactly 1.

Now, given an integer array, you need to find the length of its longest harmonious subsequence among all its possible subsequences.

Input: [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].
```
##### mine
```
class Solution {
    fun findLHS(nums: IntArray): Int {
        class Count(var left: Int = 0, var own: Int = 0, var right: Int = 0) {
            val max: Int
                get() {
                    return if (left == 0 && right == 0 && own != 0) {
                        0
                    } else if (own == 0) {
                        0
                    } else {
                        if (left > right) {
                            left + own
                        } else {
                            right + own
                        }
                    }
                }
        }

        val map = mutableMapOf<Int, Count>()
        for (num in nums) {
            if (map[num - 1] == null) map[num - 1] = Count()
            if (map[num] == null) map[num] = Count()
            if (map[num + 1] == null) map[num + 1] = Count()

            map[num - 1]!!.right++
            map[num]!!.own++
            map[num - 1]!!.left++
        }


        return map.values.map {
            it.max
        }.max() ?: 0
    }
}
```
##### like
```
class Solution {
    fun findLHS(nums: IntArray): Int {
        val map = mutableMapOf<Int, Int>()
        for (num in nums) {
            map[num] = map.getOrDefault(num, 0) + 1
        }
        var result = 0
        for (key in map.keys) {
            if (map.containsKey(key + 1)) {
                result = result.coerceAtLeast(map[key + 1]!! + map[key]!!)
            }
        }
        return result
    }
}
```
### 20
```
Patching Array

Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.

Input: nums = [1,3], n = 6
Output: 1 
Explanation:
Combinations of nums are [1], [3], [1,3], which form possible sums of: 1, 3, 4.
Now if we add/patch 2 to nums, the combinations are: [1], [2], [3], [1,3], [2,3], [1,2,3].
Possible sums are 1, 2, 3, 4, 5, 6, which now covers the range [1, 6].
So we only need 1 patch.

Input: nums = [1,5,10], n = 20
Output: 2
Explanation: The two patches can be [2, 4].

Input: nums = [1,2,2], n = 5
Output: 0
```
##### mine
```
```
##### like
```
class Solution {
    fun minPatches(nums: IntArray, n: Int): Int {
        var patch = 0
        var i = 0
        var miss = 1L
        while (miss <= n) {
            if (i >= nums.size || miss < nums[i]) {
                miss += miss
                patch++
            } else {
                miss += nums[i++]
            }
        }
        return patch
    }
}
```
### 21
```
Smallest Range Covering Elements from K Lists

You have k lists of sorted integers in ascending order. Find the smallest range that includes at least one number from each of the k lists.

We define the range [a,b] is smaller than range [c,d] if b-a < d-c or a < c if b-a == d-c.

Input: [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
Output: [20,24]
Explanation: 
List 1: [4, 10, 15, 24,26], 24 is in range [20,24].
List 2: [0, 9, 12, 20], 20 is in range [20,24].
List 3: [5, 18, 22, 30], 22 is in range [20,24].

Note
The given list may contain duplicates, so ascending order means >= here.
1 <= k <= 3500
-105 <= value of elements <= 105.
```
##### 1
```
class Solution {
    fun smallestRange(nums: List<List<Int>>): IntArray {
        val res = intArrayOf(Int.MIN_VALUE, Int.MAX_VALUE)
        val indexes = IntArray(nums.size)
        val pq = PriorityQueue<Int>(Comparator<Int> { o1, o2 ->
            nums[o1][indexes[o1]] - nums[o2][indexes[o2]]
        })

        var max = Int.MIN_VALUE
        for (i in nums.indices) {
            pq.offer(i)
            max = if (max < nums[i][0]) nums[i][0] else max
        }

        var range = Int.MAX_VALUE

        while (true) {
            val curIndex = pq.poll() ?: break
            if (range > max - nums[curIndex][indexes[curIndex]]) {
                range = max - nums[curIndex][indexes[curIndex]]
                res[1] = max
                res[0] = nums[curIndex][indexes[curIndex]]
            }
            indexes[curIndex]++
            if (indexes[curIndex] == nums[curIndex].size) {
                break
            }
            if (nums[curIndex][indexes[curIndex]] > max) {
                max = nums[curIndex][indexes[curIndex]]
            }
            pq.offer(curIndex)
        }
        return res
    }
}
```
### 22
```
Range Module

A Range Module is a module that tracks ranges of numbers. Your task is to design and implement the following interfaces in an efficient manner.

addRange(int left, int right) Adds the half-open interval [left, right), tracking every real number in that interval. Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval [left, right) that are not already tracked.

queryRange(int left, int right) Returns true if and only if every real number in the interval [left, right) is currently being tracked.

removeRange(int left, int right) Stops tracking every real number currently being tracked in the interval [left, right).
```
##### mine
```
class RangeModule() {

    private val pq = mutableListOf<IntArray>()

    fun addRange(left: Int, right: Int) {
        val iterator = pq.iterator()
        while (iterator.hasNext()) {
            val a = iterator.next()
            if (a[0] >= left && a[1] <= right) {
                iterator.remove()
            }
        }

        val f = pq.find {
            left >= it[0] && left <= it[1]
        }?.also {
            if (right <= it[1]) {
                /**
                 * range already exist
                 */
                return
            }
        }

        val l = pq.findLast {
            right >= it[0] && right <= it[1]
        }?.also {
            if (left >= it[0]) {
                /**
                 * range already exist
                 */
                return
            }
        }

        if (f == null && l == null) {
            pq.add(intArrayOf(left, right))
        } else if (f != null && l == null) {
            f[1] = right
        } else if (f == null && l != null) {
            l[0] = left
        } else if (l != null && f != null) {
            l[0] = f[0]
            pq.remove(f)
        }
    }

    fun queryRange(left: Int, right: Int): Boolean {
        return pq.find {
            it[0] <= left && it[1] >= right
        } != null
    }

    fun removeRange(left: Int, right: Int) {
        val iterator = pq.iterator()
        while (iterator.hasNext()) {
            val a = iterator.next()
            if (a[0] >= left && a[1] <= right) {
                iterator.remove()
            }
        }
        
        val f = pq.find {
            left >= it[0] && left <= it[1]
        }?.also {
            if (right <= it[1]) {
                val temp = it[1]
                it[1] = left
                pq.add(intArrayOf(right, temp))
                return
            }
        }

        val l = pq.findLast {
            right >= it[0] && right <= it[1]
        }?.also {
            if (left >= it[0]) {
                val temp = it[0]
                it[0] = right
                pq.add(intArrayOf(temp, left))
                return
            }
        }

        if (f == null && l == null) {
            return
        }

        if (f != null) {
            f[1] = left
        }

        if (l != null) {
            l[0] = right
        }
    }

}

/**
 * Your RangeModule object will be instantiated and called as such:
 * var obj = RangeModule()
 * obj.addRange(left,right)
 * var param_2 = obj.queryRange(left,right)
 * obj.removeRange(left,right)
 */
```
### 23
```
Binary Tree Pruning

We are given the head node root of a binary tree, where additionally every node's value is either a 0 or a 1.

Return the same tree where every subtree (of the given tree) not containing a 1 has been removed.

(Recall that the subtree of a node X is X, plus every node that is a descendant of X.)

Example 1:
Input: [1,null,0,0,1]
Output: [1,null,0,null,1]

Explanation: 
Only the red nodes satisfy the property "every subtree not containing a 1".
The diagram on the right represents the answer.
1     1
 0  -> 0
0 1     1

Example 2:
Input: [1,0,1,0,0,0,1]
Output: [1,null,1,null,1]
   1               1
 0   1   ->         1
0 0 0 1              1

Example 3:
Input: [1,1,0,1,1,0,1,0]
Output: [1,1,0,1,1,null,1]
     1                   1
  1    0               1   0
 1 1  0 1   ->        1 1    1
0 

Note:
The binary tree will have at most 200 nodes.
The value of each node will only be 0 or 1
```
##### mine
```
/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
class Solution {
    fun pruneTree(root: TreeNode?): TreeNode? {
        if (root == null) return null
        if (root.left != null)
            root.left = pruneTree(root.left)
        if (root.right != null)
            root.right = pruneTree(root.right)
        if (root.left == null && root.right == null) {
            return if (root.`val` == 1) root else null
        }
        return root
    }
}
```
### 24
```
Max Dot Product of Two Subsequences

Given two arrays nums1 and nums2

Return the maximum dot product between non-empty subsequences of nums1 and nums2 with the same length.

A subsequence of a array is a new array which is formed from the original array by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, [2,3,5] is a subsequence of [1,2,3,4,5] while [1,5,3] is not).

Example 1:
Input: nums1 = [2,1,-2,5], nums2 = [3,0,-6]
Output: 18
Explanation: Take subsequence [2,-2] from nums1 and subsequence [3,-6] from nums2.
Their dot product is (2*3 + (-2)*(-6)) = 18.

Example 2:
Input: nums1 = [3,-2], nums2 = [2,-6,7]
Output: 21
Explanation: Take subsequence [3] from nums1 and subsequence [7] from nums2.
Their dot product is (3*7) = 21.

Example 3:
Input: nums1 = [-1,-1], nums2 = [1,1]
Output: -1
Explanation: Take subsequence [-1] from nums1 and subsequence [1] from nums2.
Their dot product is -1.

Constraints:

1 <= nums1.length, nums2.length <= 500
-1000 <= nums1[i], nums2[i] <= 1000
```
##### dp solve
```
class Solution {
    fun maxDotProduct(nums1: IntArray, nums2: IntArray): Int {
        val temp = Array(nums1.size) {
            Array(nums2.size) {
                Int.MIN_VALUE
            }
        }
        var res = Int.MIN_VALUE
        for (i in nums1.indices) {
            for (j in nums2.indices) {
                if (i == 0 && j == 0) {
                    temp[i][j] = nums1[i] * nums2[j]
                } else if (i == 0) {
                    val a = nums1[i] * nums2[j]
                    val c = temp[i][j - 1]
                    temp[i][j] = Math.max(a, c)
                } else if (j == 0) {
                    val a = nums1[i] * nums2[j]
                    val b = temp[i - 1][j]
                    temp[i][j] = Math.max(a, b)
                } else {
                    val a = nums1[i] * nums2[j]
                    val b = temp[i - 1][j]
                    val c = temp[i][j - 1]
                    val d = temp[i - 1][j - 1] + a
                    temp[i][j] = Math.max(Math.max(a, b), Math.max(c, d))
                }
                if (res < temp[i][j])
                    res = temp[i][j]
            }
        }
        return res
    }
}
```
### 25
```
Decrypt String from Alphabet to Integer Mapping

Given a string s formed by digits ('0' - '9') and '#' . We want to map s to English lowercase characters as follows:

Characters ('a' to 'i') are represented by ('1' to '9') respectively.
Characters ('j' to 'z') are represented by ('10#' to '26#') respectively. 
Return the string formed after mapping.

It's guaranteed that a unique mapping will always exist.

Input: s = "10#11#12"
Output: "jkab"
Explanation: "j" -> "10#" , "k" -> "11#" , "a" -> "1" , "b" -> "2".

Input: s = "1326#"
Output: "acz"

Input: s = "25#"
Output: "y"

Input: s = "12345678910#11#12#13#14#15#16#17#18#19#20#21#22#23#24#25#26#"
Output: "abcdefghijklmnopqrstuvwxyz"

Constraints:

1 <= s.length <= 1000
s[i] only contains digits letters ('0'-'9') and '#' letter.
s will be valid string such that mapping is always possible.
```
##### mine
```
class Solution {
    fun freqAlphabets(s: String): String {
        val map1 = mutableMapOf<Char, String>().apply {
            put('1', "a")
            put('2', "b")
            put('3', "c")
            put('4', "d")
            put('5', "e")
            put('6', "f")
            put('7', "g")
            put('8', "h")
            put('9', "i")
        }
        val map2 = mutableMapOf<String, String>().apply {
            put("10#", "j")
            put("11#", "k")
            put("12#", "l")
            put("13#", "m")
            put("14#", "n")
            put("15#", "o")
            put("16#", "p")
            put("17#", "q")
            put("18#", "r")
            put("19#", "s")
            put("20#", "t")
            put("21#", "u")
            put("22#", "v")
            put("23#", "w")
            put("24#", "x")
            put("25#", "y")
            put("26#", "z")
        }
        var res = ""
        var i = s.length - 1
        while (i >= 0) {
            if (s[i] == '#') {
                res = map2[s.substring(i - 2, i + 1)] + res
                i -= 3
            } else {
                res = map1[s[i]] + res
                i--
            }
        }
        return res
    }
}
```
### 26
```
Reverse Nodes in k-Group

Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

Example:
Given this linked list: 1->2->3->4->5
For k = 2, you should return: 2->1->4->3->5
For k = 3, you should return: 3->2->1->4->5

Note:
Only constant extra memory is allowed.
You may not alter the values in the list's nodes, only nodes itself may be changed.
```
##### mine
```
/**
 * Example:
 * var li = ListNode(5)
 * var v = li.`val`
 * Definition for singly-linked list.
 * class ListNode(var `val`: Int) {
 *     var next: ListNode? = null
 * }
 */
class Solution {
    fun reverseKGroup(head: ListNode?, k: Int): ListNode? {
        var curRoot = head
        var preRoot = curRoot

        var preLast: ListNode? = null

        var resNode: ListNode? = null

        while (curRoot != null) {
            var count = 0
            while (curRoot != null && count < k) {
                val temp = curRoot
                curRoot = curRoot.next
                count++
                if (count == k) {
                    /**
                     * 断开准备reverse
                     */
                    temp.next = null
                }
            }
            if(count == k) {
                if (resNode == null) {
                    /**
                     * 第一次reverse
                     */
                    resNode = reverse(preRoot)
                } else {
                    /**
                     * 之后的reverse接在前一次之后
                     */
                    preLast?.next = reverse(preRoot)
                }
                /**
                 * 反转后preRoot从最前变最末尾，接上当前的curRoot继续
                 */
                preRoot?.next = curRoot
                /**
                 * preRoot反转后变为preLast
                 */
                preLast = preRoot
                /**
                 * 最后preRoot移动到curRoot准备下次反转
                 */
                preRoot = curRoot
            }
        }
        return resNode
    }
    
    fun reverse(head: ListNode?): ListNode? {
        if (head == null) return head
        var cur: ListNode? = null
        var a = head
        while (a != null) {
            val temp = a.next
            a.next = cur
            cur = a
            if (temp == null) {
                return a
            }
            a = temp
        }
        return null
    }
}
```
### 27
```
Sequential Digits

An integer has sequential digits if and only if each digit in the number is one more than the previous digit.

Return a sorted list of all the integers in the range [low, high] inclusive that have sequential digits.

Input: low = 100, high = 300
Output: [123,234]

Input: low = 1000, high = 13000
Output: [1234,2345,3456,4567,5678,6789,12345]

10 <= low <= high <= 10^9
```
##### mine
```
class Solution {
    fun sequentialDigits(low: Int, high: Int): List<Int> {
        if (low < 10 || high > 10_000_000_000) return emptyList()

        val res = mutableListOf<Int>()

        var temp = ""

        for (i in '1'..'8') {
            temp += i
            var j = 1
            temp += (i + j)
            var cur = temp.toInt()
            while (cur < high) {
                if (cur >= low) {
                    res.add(cur)
                }
                if (i + j == '9') {
                    break
                }
                temp += (i + ++j)
                cur = temp.toInt()
            }

            temp = ""
        }

        return res.sorted()
    }
}
```
### 28
```
Maximum Students Taking Exam

Given a m * n matrix seats  that represent seats distributions in a classroom. If a seat is broken, it is denoted by '#' character otherwise it is denoted by a '.' character.

Students can see the answers of those sitting next to the left, right, upper left and upper right, but he cannot see the answers of the student sitting directly in front or behind him. Return the maximum number of students that can take the exam together without any cheating being possible..

Students must be placed in seats in good condition.
```
##### choose
```
fun maxStudents(seats: Array<CharArray>): Int {
        val m: Int = seats.size
        val n: Int = seats[0].size
        val validity = IntArray(m) // validity数组用于记录每一横排位置是否能坐
        val stateSize = 1 shl n // 每一横排可由学生排布的方式有2^n种
        val dp = Array(m) { IntArray(stateSize) }
        var ans = 0
        // 初始化validity数组
        for (i in 0 until m) {
            for (j in 0 until n) {
                validity[i] = (validity[i] shl 1) + if (seats[i][j] == '.') 1 else 0
            }
        }
        // 初始化dp数组
        for (i in 0 until m) {
            for (j in 0 until stateSize) {
                dp[i][j] = -1
            }
        }

        for (i in 0 until m) {
            for (j in 0 until stateSize) {
                // j & validity[i] == j 判断j的状态下能否坐下第i横排
                // (j & (j >> 1) == 0) 判断j模式左右是否没人
                if (j and validity[i] == j && j and (j shr 1) == 0) {
                    if (i == 0) { // 第一横排
                        dp[i][j] = Integer.bitCount(j)
                    } else {
                        // 不是第一排，就要遍历前一排，从而取得当前排的最大值。
                        for (k in 0 until stateSize) {
                            if (j and (k shr 1) == 0 && j shr 1 and k == 0 && dp[i - 1][k] != -1) {
                                dp[i][j] = max(dp[i - 1][k] + Integer.bitCount(j), dp[i][j])
                            }
                        }
                    }
                    ans = max(ans, dp[i][j])
                }
            }
        }
        return ans
    }
```
### 29
```
Maximize Sum Of Array After K Negations

Given an array A of integers, we must modify the array in the following way: we choose an i and replace A[i] with -A[i], and we repeat this process K times in total.  (We may choose the same index i multiple times.)

Return the largest possible sum of the array after modifying it in this way.


Input: A = [4,2,3], K = 1
Output: 5
Explanation: Choose indices (1,) and A becomes [4,-2,3].


Input: A = [3,-1,0,2], K = 3
Output: 6
Explanation: Choose indices (1, 2, 2) and A becomes [3,1,0,2].


Input: A = [2,-3,-1,5,-4], K = 2
Output: 13
Explanation: Choose indices (1, 4) and A becomes [2,3,-1,5,4].


Note:
1 <= A.length <= 10000
1 <= K <= 10000
-100 <= A[i] <= 100
```
##### mine
```
class Solution {
    fun largestSumAfterKNegations(A: IntArray, K: Int): Int {
        if (A.isEmpty()) return 0

        val B = A.filter {
            it < 0
        }.sorted()

        val C = A.map {
            Math.abs(it)
        }.sorted().toMutableList()

        return if (B.size == K) C.sum() else if (B.size > K) C.sum() + B.subList(K, B.size)
            .sum() * 2 else {
            if ((K - B.size) % 2 == 0) C.sum() else C.apply { C[0] = -C[0] }.sum()
        }
    }
}
```
##### best
```
class Solution {
    fun largestSumAfterKNegations(A: IntArray, K: Int): Int {
        val pq = PriorityQueue<Int>()

        for (x in A) pq.add(x)
        var count = K
        while (count-- > 0) pq.add(-pq.poll()!!)

        var sum = 0
        for (i in A.indices) {
            sum += pq.poll()!!
        }
        return sum
    }
}
```
### 30
```
Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Example 2:
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4

Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.
```
##### mine
```
class Solution {
    fun findKthLargest(nums: IntArray, k: Int): Int {
        val pq = PriorityQueue<Int>()
        
        for(num in nums) pq.add(-num)
        var count = k
        while (count > 1) {
            pq.poll()
            count--
        }
        return -pq.poll()!!
    }
}
```
##### like
```
class Solution {
    fun findKthLargest(nums: IntArray, k: Int): Int {
        return nums.apply { sort() }[nums.size-k]
    }
}
```
### 31
```
Maximum Number of Non-Overlapping Substrings

Given a string s of lowercase letters, you need to find the maximum number of non-empty substrings of s that meet the following conditions:

The substrings do not overlap, that is for any two substrings s[i..j] and s[k..l], either j < k or i > l is true.
A substring that contains a certain character c must also contain all occurrences of c.
Find the maximum number of substrings that meet the above conditions. If there are multiple solutions with the same number of substrings, return the one with minimum total length. It can be shown that there exists a unique solution of minimum total length.

Notice that you can return the substrings in any order.

Example 1:
Input: s = "adefaddaccc"
Output: ["e","f","ccc"]
Explanation: The following are all the possible substrings that meet the conditions:
[
  "adefaddaccc"
  "adefadda",
  "ef",
  "e",
  "f",
  "ccc",
]
If we choose the first string, we cannot choose anything else and we'd get only 1. If we choose "adefadda", we are left with "ccc" which is the only one that doesn't overlap, thus obtaining 2 substrings. Notice also, that it's not optimal to choose "ef" since it can be split into two. Therefore, the optimal way is to choose ["e","f","ccc"] which gives us 3 substrings. No other solution of the same number of substrings exist.

Example 2:
Input: s = "abbaccd"
Output: ["d","bb","cc"]
Explanation: Notice that while the set of substrings ["d","abba","cc"] also has length 3, it's considered incorrect since it has larger total length.

Constraints:
1 <= s.length <= 10^5
s contains only lowercase English letters.
```
##### anwser
```
    fun maxNumOfSubstrings(s: String): List<String> {
        val seg = arrayOfNulls<Seg>(26)
        for (i in 0..25) {
            seg[i] = Seg(-1, -1)
        }
        // 预处理左右端点
        for (i in s.indices) {
            val charIdx: Int = s[i] - 'a'
            if (seg[charIdx]!!.left == -1) {
                seg[charIdx]!!.right = i
                seg[charIdx]!!.left = seg[charIdx]!!.right
            } else {
                seg[charIdx]!!.right = i
            }
        }
        for (i in 0..25) {
            if (seg[i]!!.left != -1) {
                var j = seg[i]!!.left
                while (j <= seg[i]!!.right) {
                    val charIdx: Int = s[j] - 'a'
                    if (seg[i]!!.left <= seg[charIdx]!!.left && seg[charIdx]!!.right <= seg[i]!!.right
                    ) {
                        ++j
                        continue
                    }
                    seg[i]!!.left = Math.min(seg[i]!!.left, seg[charIdx]!!.left)
                    seg[i]!!.right = Math.max(seg[i]!!.right, seg[charIdx]!!.right)
                    j = seg[i]!!.left
                    ++j
                }
            }
        }
        // 贪心选取
        Arrays.sort(seg)
        val ans: MutableList<String> = ArrayList()
        var end = -1
        for (segment in seg) {
            val left = segment!!.left
            val right = segment!!.right
            if (left == -1) {
                continue
            }
            if (end == -1 || left > end) {
                end = right
                ans.add(s.substring(left, right + 1))
            }
        }
        return ans
    }
```
### 32
```
Random Pick Index

Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Note:
The array size can be very large. Solution that uses too much extra space will not pass the judge.

Example:
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);

// pick(3) should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(3);

// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
```
##### mine
```
class Solution(val nums: IntArray) {
    val random = Random(System.currentTimeMillis())

    fun pick(target: Int): Int {
      var count = 0
            for (num in nums) {
                if (target == num) count++
            }

            if (count == 0) return -1

            val random = (random.nextDouble() * count).toInt()
            count = 0
            for ((index, num) in nums.withIndex()) {
                if (target == num) {
                    if (count == random) {
                        return index
                    } else {
                        count++
                    }
                }
            }
            return nums.indexOf(target)
    }

}

/**
 * Your Solution object will be instantiated and called as such:
 * var obj = Solution(nums)
 * var param_1 = obj.pick(target)
 */
```
### 33
```
Find Latest Group of Size M

Given an array arr that represents a permutation of numbers from 1 to n. You have a binary string of size n that initially has all its bits set to zero.

At each step i (assuming both the binary string and arr are 1-indexed) from 1 to n, the bit at position arr[i] is set to 1. You are given an integer m and you need to find the latest step at which there exists a group of ones of length m. A group of ones is a contiguous substring of 1s such that it cannot be extended in either direction.

Return the latest step at which there exists a group of ones of length exactly m. If no such group exists, return -1.

Input: arr = [3,5,1,2,4], m = 1
Output: 4
Explanation:
Step 1: "00100", groups: ["1"]
Step 2: "00101", groups: ["1", "1"]
Step 3: "10101", groups: ["1", "1", "1"]
Step 4: "11101", groups: ["111", "1"]
Step 5: "11111", groups: ["11111"]
The latest step at which there exists a group of size 1 is step 4.

Input: arr = [3,1,5,4,2], m = 2
Output: -1
Explanation:
Step 1: "00100", groups: ["1"]
Step 2: "10100", groups: ["1", "1"]
Step 3: "10101", groups: ["1", "1", "1"]
Step 4: "10111", groups: ["1", "111"]
Step 5: "11111", groups: ["11111"]
No group of size 2 exists during any step.

Input: arr = [1], m = 1
Output: 1

Input: arr = [2,1], m = 2
Output: 2

Constraints:
n == arr.length
1 <= n <= 10^5
1 <= arr[i] <= n
All integers in arr are distinct.
1 <= m <= arr.length
```
##### like
```
class Solution {
    fun findLatestStep(arr: IntArray, m: Int): Int {
        var res = -1
        val n = arr.size
        if (n == m) return n
        val length = IntArray(n + 2)
        for (i in 0 until n) {
            val a = arr[i]
            val left = length[a - 1]
            val right = length[a + 1]
            length[a + right] = left + right + 1
            length[a - left] = length[a + right]
            if (left == m || right == m) res = i
        }
        return res
    }
}
```
### 34
```
Second Highest Salary

Write a SQL query to get the second highest salary from the Employee table.
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
For example, given the above Employee table, the query should return 200 as the second highest salary. If there is no second highest salary, then the query should return null.
+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
```
##### anwser
```
SELECT max(Salary) as SecondHighestSalary
FROM Employee
WHERE Salary < (SELECT max(Salary) FROM Employee)
```
### 35
```
Best Time to Buy and Sell Stock III

Say you have an array for which the i'th element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

Example 4:
Input: prices = [1]
Output: 0

Constraints:
1 <= prices.length <= 10'5
0 <= prices[i] <= 10'5
```
##### like
```
class Solution {
    fun maxProfit(prices: IntArray): Int {
        var cost1 = Int.MAX_VALUE
        var cost2 = Int.MAX_VALUE
        var pro1 = 0
        var pro2 = 0
        for(i in prices) {
            cost1 = Math.min(cost1, i)
            pro1 = Math.max(pro1, i - cost1)
            cost2 = Math.min(cost2, i - pro1)
            pro2 = Math.max(pro2, i - cost2)
        }
        return pro2
    }
}
```
### 36
```
Find Smallest Letter Greater Than Target

Given a list of sorted characters letters containing only lowercase letters, and given a target letter target, find the smallest element in the list that is larger than the given target.

Letters also wrap around. For example, if the target is target = 'z' and letters = ['a', 'b'], the answer is 'a'.

Input:
letters = ["c", "f", "j"]
target = "a"
Output: "c"

Input:
letters = ["c", "f", "j"]
target = "c"
Output: "f"

Input:
letters = ["c", "f", "j"]
target = "d"
Output: "f"

Note:
letters has a length in range [2, 10000].
letters consists of lowercase letters, and contains at least 2 unique letters.
target is a lowercase letter.
```
##### mine
```
class Solution {
    fun nextGreatestLetter(letters: CharArray, target: Char): Char {
        var min = Int.MAX_VALUE
        var resIndex = -1
        for((index, l) in letters.withIndex()) {
            if(l == target) continue
            val res = if(l > target) {
                l - target
            } else {
                l + 26 - target
            }
            if(res > 0) {
                if(min > res) {
                    min = res
                    resIndex = index
                }
            }
        }
        return letters[resIndex]
    }
}
```
##### like
```
class Solution {
    fun nextGreatestLetter(letters: CharArray, target: Char): Char {
        val n: Int = letters.size

        //hi starts at 'n' rather than the usual 'n - 1'. 
        //It is because the terminal condition is 'lo < hi' and if hi starts from 'n - 1', 
        //we can never consider value at index 'n - 1'
        var lo = 0
        var hi = n

        //Terminal condition is 'lo < hi', to avoid infinite loop when target is smaller than the first element
        while (lo < hi) {
            val mid = lo + (hi - lo) / 2
            if (letters[mid] > target) 
                hi = mid 
            else 
                lo = mid + 1 //letters[mid] <= target
        }
        //Because lo can end up pointing to index 'n', in which case we return the first element
        return letters[lo % n]
    }
}
```
### 37
```
Longest Well-Performing Interval

We are given hours, a list of the number of hours worked per day for a given employee.

A day is considered to be a tiring day if and only if the number of hours worked is (strictly) greater than 8.

A well-performing interval is an interval of days for which the number of tiring days is strictly larger than the number of non-tiring days.

Return the length of the longest well-performing interval.

Input: hours = [9,9,6,0,6,6,9]
Output: 3
Explanation: The longest well-performing interval is [9,9,6].

Constraints:
1 <= hours.length <= 10000
0 <= hours[i] <= 16
```
##### mine
```
class Solution {
    fun longestWPI(hours: IntArray): Int {
        for(i in hours.indices){
            hours[i] = if(hours[i] - 8 > 0) 1 else -1
        }

        val p = IntArray(hours.size + 1)
        var cur = 0
        for(i in hours.indices) {
            p[i] = cur
            cur += hours[i]
        }
        p[p.size - 1] = cur

        var res = 0
        for(i in p.indices) {
            for(j in i + 1 until p.size) {
                if(p[i] < p[j]) res = Math.max(res, j - i)
            }
        }

        return res
    }
}
```
### 38
```
Maximum Number of Occurrences of a Substring

Given a string s, return the maximum number of ocurrences of any substring under the following rules:
    The number of unique characters in the substring must be less than or equal to maxLetters.
    The substring size must be between minSize and maxSize inclusive.

Example 1:
Input: s = "aababcaab", maxLetters = 2, minSize = 3, maxSize = 4
Output: 2
Explanation: Substring "aab" has 2 ocurrences in the original string.
It satisfies the conditions, 2 unique letters and size 3 (between minSize and maxSize).

Example 2:
Input: s = "aaaa", maxLetters = 1, minSize = 3, maxSize = 3
Output: 2
Explanation: Substring "aaa" occur 2 times in the string. It can overlap.

Example 3:
Input: s = "aabcabcab", maxLetters = 2, minSize = 2, maxSize = 3
Output: 3

Example 4:
Input: s = "abcde", maxLetters = 2, minSize = 3, maxSize = 3
Output: 0

Constraints:
1 <= s.length <= 10^5
1 <= maxLetters <= 26
1 <= minSize <= maxSize <= min(26, s.length)
s only contains lowercase English letters.
```
##### mine Time Limit Exceeded
```
class Solution {
    fun maxFreq(s: String, maxLetters: Int, minSize: Int, maxSize: Int): Int {
        var res = 0
        val set = mutableSetOf<Char>()
        for (i in s.indices) {
            for (j in i until s.length) {
                if (j - i in minSize..maxSize) {
                    val child = s.substring(i, j)
                    set.clear()
                    var needNext = true
                    for (c in child) {
                        set.add(c)
                        if (set.size > maxLetters) {
                            needNext = false
                            break
                        }
                    }
                    if (needNext) {
                        val count = count(s, child)
                        if (count > res) {
                            res = count
                        }
                    } else {
                        break
                    }
                }
            }
        }
        return res
    }
    
    fun count(s: String, child: String): Int {
        var i = 0
        var count = 0
        while (s.length - i >= child.length) {
            val firstIndex = s.indexOf(child, i)
            if (firstIndex != -1) {
                count++
                i = firstIndex + 1
            } else {
                break
            }
        }
        return count
    }
}
```
##### mine 2
```
class Solution {
    fun maxFreq(s: String, maxLetters: Int, minSize: Int, maxSize: Int): Int {
        var res = 0
        val map = mutableMapOf<String, Int>()
        val set = mutableSetOf<Char>()
        for (i in s.indices) {
            val end = Math.min(s.length, i + maxSize)
            for (j in i..end) {
                if (j - i in minSize..maxSize) {
                    val child = s.substring(i, j)
                    var needNext = true
                    set.clear()
                    for (c in child) {
                        set.add(c)
                        if (set.size > maxLetters) {
                            needNext = false
                            break
                        }
                    }
                    if (needNext) {
                        if (map.containsKey(child)) {
                            map[child] = map[child]!! + 1
                        } else {
                            map[child] = 1
                        }
                        res = Math.max(res, map[child]!!)
                    }
                }
            }
        }
        return res
    }
}
```
##### mine 3
```
class Solution {
    fun maxFreq(s: String, maxLetters: Int, minSize: Int, maxSize: Int): Int {
        var res = 0
        val map = mutableMapOf<String, Int>()
        val set = mutableSetOf<Char>()
        for (i in 0..(s.length - minSize)) {
            val child = s.substring(i, i + minSize)
            var needNext = true
            set.clear()
            for (c in child) {
                set.add(c)
                if (set.size > maxLetters) {
                    needNext = false
                    break
                }
            }

            if(needNext) {
                if (map.containsKey(child)) {
                    map[child] = map[child]!! + 1
                } else {
                    map[child] = 1
                }
                res = Math.max(res, map[child]!!)
            }
        }
        return res
    }
}
```
### 39
```
Print in Order

Suppose we have a class:
public class Foo {
  public void first() { print("first"); }
  public void second() { print("second"); }
  public void third() { print("third"); }
}

The same instance of Foo will be passed to three different threads. Thread A will call first(), thread B will call second(), and thread C will call third(). Design a mechanism and modify the program to ensure that second() is executed after first(), and third() is executed after second().
```
##### mine
```
class Foo {

    public Foo() {
        
    }
    
    volatile boolean isFirstRun = false;
    volatile boolean isSecondRun = false;
    private final Object lock = new Object();

    public void first(Runnable printFirst) throws InterruptedException {
        synchronized(lock) {
            // printFirst.run() outputs "first". Do not change or remove this line.
            printFirst.run();
            isFirstRun = true;
            lock.notifyAll();
        }
    }

    public void second(Runnable printSecond) throws InterruptedException {
        synchronized(lock) {
            while(!isFirstRun){
                lock.wait();
            }
            // printSecond.run() outputs "second". Do not change or remove this line.
            printSecond.run();
            isSecondRun = true;
            lock.notifyAll();
        }
        
    }

    public void third(Runnable printThird) throws InterruptedException {
        synchronized(lock) { 
            while(!isSecondRun){
                lock.wait();
            }
            // printThird.run() outputs "third". Do not change or remove this line.
            printThird.run();
        }
    }
}
```
### 40
```
Reorder List

Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:
Given 1->2->3->4, reorder it to 1->4->2->3.

Example 2:
Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
```
##### mine
```
/**
 * Example:
 * var li = ListNode(5)
 * var v = li.`val`
 * Definition for singly-linked list.
 * class ListNode(var `val`: Int) {
 *     var next: ListNode? = null
 * }
 */
class Solution {
    fun reorderList(head: ListNode?): Unit {
        if(head == null) return

        var cur = head!!
        val link = LinkedList<ListNode>()
        while (cur.next != null) {
            link.add(cur.next!!)
            cur.next = cur.next!!.next
        }
        var i = 0
        while (link.size > 0) {
            if(i % 2 == 0) {
                cur.next = link.removeLast()
            } else {
                cur.next = link.removeFirst()
            }
            cur = cur.next!!
            if(link.size == 0) {
                cur.next = null
            }
            i++
        }
    }
}
```
### 41
```
Permutation in String

Given two strings s1 and s2, write a function to return true if s2 contains the permutation of s1. In other words, one of the first string's permutations is the substring of the second string.

Example 1:
Input: s1 = "ab" s2 = "eidbaooo"
Output: True
Explanation: s2 contains one permutation of s1 ("ba").

Example 2:
Input:s1= "ab" s2 = "eidboaoo"
Output: False

Constraints:
The input strings only contain lower case letters.
The length of both given strings is in range [1, 10,000].
```
##### mine too slow
```
class Solution {
    fun checkInclusion(s1: String, s2: String): Boolean {
        if (s1.length > s2.length) return false
        if (s2.contains(s1)) return true

        val targetMap = mutableMapOf<Char, Int>()
        for (c in s1) {
            if (targetMap.containsKey(c)) {
                targetMap[c] = targetMap[c]!! + 1
            } else {
                targetMap[c] = 1
            }
        }

        val tempMap = mutableMapOf<Char, Int>()
        for (i in 0..(s2.length - s1.length)) {
            tempMap.clear()
            for (j in i until (i + s1.length)) {
                if (tempMap.containsKey(s2[j])) {
                    tempMap[s2[j]] = tempMap[s2[j]]!! + 1
                } else {
                    tempMap[s2[j]] = 1
                }
            }

            if (tempMap.size == targetMap.size) {
                var contain = true
                tempMap.forEach {
                    if (it.value != targetMap[it.key]) {
                        contain = false
                        return@forEach
                    }
                }

                if (contain) {
                    return true
                }
            }
        }
        return false
    }
}
```
##### like
```
class Solution {
    fun checkInclusion(s1: String, s2: String): Boolean {
        if (s1.length > s2.length) return false

        val count = IntArray(26)
        for (i in s1.indices) {
            count[s1[i] - 'a']++
            count[s2[i] - 'a']--
        }
        if (allZero(count)) return true

        for (i in s1.length until s2.length) {
            count[s2[i - s1.length] - 'a']++
            count[s2[i] - 'a']--
            if (allZero(count)) return true
        }

        return false
    }

    fun allZero(array: IntArray): Boolean {
        for (a in array) {
            if (a != 0) return false
        }
        return true
    }
}
```
### 42
```
Total Hamming Distance

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Now your job is to find the total Hamming distance between all pairs of the given numbers.

Example:
Input: 4, 14, 2
Output: 6
Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
showing the four bits relevant in this case). So the answer will be:
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.

Note:
Elements of the given array are in the range of 0 to 10^9
Length of the array will not exceed 10^4.
```
##### mine Time Limit Exceeded
```
class Solution {
   fun totalHammingDistance(nums: IntArray): Int {
        var res = 0
        for (i in nums.indices) {
            for (j in nums.indices) {
                if (i != j) {
                    res += hammingDistance(nums[i], nums[j])
                }
            }
        }
        return res / 2
    }

    fun hammingDistance(a: Int, b: Int): Int {
        return Integer.toBinaryString(a.xor(b)).count {
            it == '1'
        }
    }
}
```
##### best
```
class Solution {
   fun totalHammingDistance(nums: IntArray): Int {
        var res = 0
        for (i in 0..31) {
            var ones = 0
            for (j in nums.indices) {
                ones += nums[j].and(1)
                nums[j] = nums[j].shr(1)
            }
            res += ones * (nums.size - ones)
        }
        return res
    }
}
```
### 43
```
Find the Difference

You are given two strings s and t.
String t is generated by random shuffling string s and then add one more letter at a random position.
Return the letter that was added to t.

Example 1:
Input: s = "abcd", t = "abcde"
Output: "e"
Explanation: 'e' is the letter that was added.

Example 2:
Input: s = "", t = "y"
Output: "y"

Example 3:
Input: s = "a", t = "aa"
Output: "a"

Example 4:
Input: s = "ae", t = "aea"
Output: "a"

Constraints:
0 <= s.length <= 1000
t.length == s.length + 1
s and t consist of lower-case English letters.
```
##### mine
```
class Solution {
    fun findTheDifference(s: String, t: String): Char {
        val ta = IntArray(26)
        for (c in t) {
            ta[c - 'a']++
        }
        for (c in s) {
            ta[c - 'a']--
        }
        return 'a' + ta.indexOf(1)
    }
}
```
### 44
```
Convert Sorted Array to Binary Search Tree

Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:
Given the sorted array: [-10,-3,0,5,9],
One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:
      0
     / \
   -3   9
   /   /
 -10  5
```
##### mine
```
/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
class Solution {
    fun sortedArrayToBST(nums: IntArray): TreeNode? {
        if(nums.isEmpty()) return null
        val middle = nums.size / 2
        return TreeNode(nums[middle]).apply {
            left = getNode(nums, 0, middle - 1)
            right = getNode(nums, middle + 1, nums.size - 1)
        }
    }

    fun getNode(nums: IntArray, startIndex: Int, endIndex: Int): TreeNode? {
        if (startIndex > endIndex) return null
        if (startIndex == endIndex) return TreeNode(nums[startIndex])
        if (startIndex < endIndex) {
            val middle = startIndex + (endIndex - startIndex) / 2
            return TreeNode(nums[middle]).apply {
                left = getNode(nums, startIndex, middle - 1)
                right = getNode(nums, middle + 1, endIndex)
            }
        }
        return null
    }
}
```
### 45
```
Water and Jug Problem

You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need to determine whether it is possible to measure exactly z litres using these two jugs.

If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end.

Operations allowed:
Fill any of the jugs completely with water.
Empty any of the jugs.
Pour water from one jug into another till the other jug is completely full or the first jug itself is empty.

Example 1: (From the famous "Die Hard" example)
Input: x = 3, y = 5, z = 4
Output: True

Example 2:
Input: x = 2, y = 6, z = 5
Output: False

Constraints:
0 <= x <= 10^6
0 <= y <= 10^6
0 <= z <= 10^6
```
##### shit
```
class Solution {
    fun canMeasureWater(x: Int, y: Int, z: Int): Boolean {
       if (x == z || y == z) return true
        if (z == x + y) return true
        if (z > x + y) return false

        return z % gcd(x, y) == 0 
    }
    
    fun gcd(a: Int, b: Int): Int {
        return if (b > 0) gcd(b, a % b) else a
    }
}
```
### 46
```
As Far from Land as Possible

Given an N x N grid containing only values 0 and 1, where 0 represents water and 1 represents land, find a water cell such that its distance to the nearest land cell is maximized and return the distance.

The distance used in this problem is the Manhattan distance: the distance between two cells (x0, y0) and (x1, y1) is |x0 - x1| + |y0 - y1|.

If no land or water exists in the grid, return -1.

Example 1:
Input: [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
Explanation: 
The cell (1, 1) is as far as possible from all the land with distance 2.

Example 2:
Input: [[1,0,0],[0,0,0],[0,0,0]]
Output: 4
Explanation: 
The cell (2, 2) is as far as possible from all the land with distance 4.

Note:
1 <= grid.length == grid[0].length <= 100
grid[i][j] is 0 or 1
```
##### mine
```
class Solution {
    fun maxDistance(grid: Array<IntArray>): Int {
        var res = 0
        val arrivedGrid = Array(grid.size) {
            BooleanArray(grid[0].size)
        }
        val queue = LinkedBlockingQueue<IntArray>()
        for (i in grid.indices) {
            for (j in grid[i].indices) {
                if (grid[i][j] == 1) {
                    arrivedGrid[i][j] = true
                    queue.add(intArrayOf(i, j, 0))
                }
            }
        }
        while (queue.isNotEmpty()) {
            val cur = queue.poll()!!
            if (cur[0] - 1 > -1 && !arrivedGrid[cur[0] - 1][cur[1]]) {
                arrivedGrid[cur[0] - 1][cur[1]] = true
                queue.add(intArrayOf(cur[0] - 1, cur[1], cur[2] + 1))
            }
            if (cur[0] + 1 < grid[0].size && !arrivedGrid[cur[0] + 1][cur[1]]) {
                arrivedGrid[cur[0] + 1][cur[1]] = true
                queue.add(intArrayOf(cur[0] + 1, cur[1], cur[2] + 1))
            }
            if (cur[1] - 1 > -1 && !arrivedGrid[cur[0]][cur[1] - 1]) {
                arrivedGrid[cur[0]][cur[1] - 1] = true
                queue.add(intArrayOf(cur[0], cur[1] - 1, cur[2] + 1))
            }
            if (cur[1] + 1 < grid.size && !arrivedGrid[cur[0]][cur[1] + 1]) {
                arrivedGrid[cur[0]][cur[1] + 1] = true
                queue.add(intArrayOf(cur[0], cur[1] + 1, cur[2] + 1))
            }
            res = Math.max(cur[2], res)
        }
        return if(res == 0) -1 else res
    }
}
```
### 47
```
Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [0]
Output: 0

Example 4:
Input: nums = [-1]
Output: -1

Example 5:
Input: nums = [-2147483647]
Output: -2147483647

Constraints:
1 <= nums.length <= 2 * 10^4
-2^31 <= nums[i] <= 2^31 - 1
```
##### mine
```
class Solution {
    fun maxSubArray(nums: IntArray): Int {
        var res = Int.MIN_VALUE
        var cur = 0
        val temp = IntArray(nums.size + 1)
        temp[0] = cur
        for (i in 1 until temp.size) {
            cur += nums[i - 1]
            temp[i] = cur
        }
        for (i in temp.indices) {
            for (j in i until temp.size) {
                if (i != j) {
                    res = Math.max(res, temp[j] - temp[i])
                }
            }
        }
        return res
    }
}
```
##### mine 2
```
class Solution {
    fun maxSubArray(nums: IntArray): Int {
        var res = nums[0]
        var sum = 0
        for (num in nums) {
            sum = if (sum > 0) {
                sum + num
            } else {
                num
            }
            res = Math.max(res, sum)
        }
        return res
    }
}
```
#####
```
class Solution {
    fun maxSubArray(nums: IntArray): Int {
        var cur = 0
        var res = nums[0]
        for(num in nums) {
            cur = Math.max(num, cur + num)
            res = Math.max(cur, res)
        }
        return res
    }
}
```
### 48
```
2 Keys Keyboard

Initially on a notepad only one character 'A' is present. You can perform two operations on this notepad for each step:
Copy All: You can copy all the characters present on the notepad (partial copy is not allowed).
Paste: You can paste the characters which are copied last time.

Given a number n. You have to get exactly n 'A' on the notepad by performing the minimum number of steps permitted. Output the minimum number of steps to get n 'A'.

Example 1:
Input: 3
Output: 3
Explanation:
Intitally, we have one character 'A'.
In step 1, we use Copy All operation.
In step 2, we use Paste operation to get 'AA'.
In step 3, we use Paste operation to get 'AAA'.

Note:
The n will be in the range [1, 1000].
```
##### mine
```
class Solution {
    fun minSteps(n: Int): Int {
        if(n == 1) return 0
        val array = IntArray((n + 1) / 2)
        for (i in 1..((n + 1) / 2)) {
            if (n % i == 0) {
                array[i - 1] = n / i
            }
        }
        var res = Int.MAX_VALUE
        for ((index, a) in array.withIndex()) {
            if (a != 0) {
                res = if (a == n) {
                    Math.min(res, a)
                } else {
                    Math.min(res, minSteps(index + 1) + a)
                }
            }
        }
        return res
    }
}
```
### 49
```
132 Pattern

Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].

Return true if there is a 132 pattern in nums, otherwise return false.

Example 1:
Input: nums = [1,2,3,4]
Output: false
Explanation: There is no 132 pattern in the sequence.

Example 2:
Input: nums = [3,1,4,2]
Output: true
Explanation: There is a 132 pattern in the sequence: [1, 4, 2].

Example 3:
Input: nums = [-1,3,2,0]
Output: true
Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].

Constraints:
n == nums.length
1 <= n <= 3 * 104
-109 <= nums[i] <= 109
```
##### mine Time Limit Exceeded
```
class Solution {
    fun find132pattern(nums: IntArray): Boolean {
        if (nums.size < 3) return false

        for (i in 0 until (nums.size - 2)) {
            for (j in i + 1 until (nums.size - 1)) {
                for (k in j + 1 until nums.size) {
                    if (nums[i] < nums[j] && nums[i] < nums[k] && nums[j] > nums[k]) {
                        return true
                    }
                }
            }
        }
        return false
    }
}
```
##### mine 2 Time Limit Exceeded
```
class Solution {
    fun find132pattern(nums: IntArray): Boolean {
        if (nums.size < 3) return false
        for (i in 1 until nums.size - 1) {
            for (s in 0 until i) {
                for (e in (i + 1) until nums.size) {
                    if (nums[i] > nums[s] && nums[i] > nums[e] && nums[e] > nums[s]) {
                        return true
                    }
                }
            }
        }
        return false
    }
}
```
##### like
```
class Solution {
    fun find132pattern(nums: IntArray): Boolean {
        if (nums.size < 3) return false
        val stack = ArrayDeque<Int>()
        val mins = IntArray(nums.size)
        mins[0] = nums[0]
        for (i in 1 until nums.size) {
            mins[i] = Math.min(mins[i - 1], nums[i])
        }
        for (i in nums.indices.reversed()) {
            if (nums[i] > mins[i]) {
                while (!stack.isEmpty() && stack.peek()!! <= mins[i]){
                    stack.pop()
                }
                if(!stack.isEmpty() && stack.peek()!! < nums[i]){
                    return true
                }
                stack.push(nums[i])
            }
        }
        return false
    }
}
```
### 50
```
Wildcard Matching

Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

Note:
s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like ? or *.

Example 1:
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Example 2:
Input:
s = "aa"
p = "*"
Output: true
Explanation: '*' matches any sequence.

Example 3:
Input:
s = "cb"
p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.

Example 4:
Input:
s = "adceb"
p = "*a*b"
Output: true
Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".

Example 5:
Input:
s = "acdcb"
p = "a*c?b"
Output: false
```
##### mine 
```
class Solution {
    fun isMatch(s: String, p: String): Boolean {
        if (s.isEmpty() && p.isEmpty()) return true
        if (s.isEmpty()) return p.count { it == '*' } == p.length
        if (p.isEmpty()) return false

        val temp = Array<Array<Boolean>>(s.length) {
            Array(p.length) {
                false
            }
        }

        temp[0][0] = s[0] == p[0] || p[0] == '*' || p[0] == '?'
        var minCount = if(s[0] == p[0] || p[0] == '?') 1 else 0
        for (i in 1 until p.length) {
            temp[0][i] = when {
                p[i] == '*' -> temp[0][i - 1]
                p[i] == '?' -> {
                    when {
                        p[i - 1] == '*' && minCount < 1 -> {
                            minCount++
                            temp[0][i - 1]
                        }
                        else -> {
                            minCount++
                            false
                        }
                    }
                }
                else -> {
                    when {
                        p[i - 1] == '*' && s[0] == p[i] && minCount < 1 -> {
                            minCount++
                            temp[0][i - 1]
                        }
                        else -> {
                            minCount++
                            false
                        }
                    }
                }
            }
        }

        for (i in 1 until s.length) {
            temp[i][0] = when {
                p[0] == '*' -> true
                else -> false
            }
        }

        for (i in 1 until s.length) {
            for (j in 1 until p.length) {
                temp[i][j] = when {
                    s[i] == p[j] -> temp[i - 1][j - 1]
                    p[j] == '?' -> temp[i - 1][j - 1]
                    p[j] == '*' -> temp[i - 1][j] || temp[i][j - 1]
                    else -> false
                }
            }
        }

        return temp[s.length - 1][p.length - 1]
    }
}
```
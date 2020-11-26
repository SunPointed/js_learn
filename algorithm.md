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
### 51
```
Number of Closed Islands

Given a 2D grid consists of 0s (land) and 1s (water).  An island is a maximal 4-directionally connected group of 0s and a closed island is an island totally (all left, top, right, bottom) surrounded by 1s.

Return the number of closed islands.

Example 1:
Input: grid = [
[1,1,1,1,1,1,1,0],
[1,0,0,0,0,1,1,0],
[1,0,1,0,1,1,1,0],
[1,0,0,0,0,1,0,1],
[1,1,1,1,1,1,1,0]]
Output: 2
Explanation: 
Islands in gray are closed because they are completely surrounded by water (group of 1s).

Example 2:
Input: grid = [
[0,0,1,0,0],
[0,1,0,1,0],
[0,1,1,1,0]]
Output: 1

Example 3:
Input: grid = [[1,1,1,1,1,1,1],
               [1,0,0,0,0,0,1],
               [1,0,1,1,1,0,1],
               [1,0,1,0,1,0,1],
               [1,0,1,1,1,0,1],
               [1,0,0,0,0,0,1],
               [1,1,1,1,1,1,1]]
Output: 2

[
[0,0,1,1,0,1,0,0,1,0],
[1,1,0,1,1,0,1,1,1,0],
[1,0,1,1,1,0,0,1,1,0],
[0,1,1,0,0,0,0,1,0,1],
[0,0,0,0,0,0,1,1,1,0],
[0,1,0,1,0,1,0,1,1,1],
[1,0,1,0,1,1,0,0,0,1],
[1,1,1,1,1,1,0,0,0,0],
[1,1,1,0,0,1,0,1,0,1],
[1,1,1,0,1,1,0,1,1,0]]

Constraints:
1 <= grid.length, grid[0].length <= 100
0 <= grid[i][j] <=1
```
##### mine
```
class Solution {
    fun closedIsland(grid: Array<IntArray>): Int {
        if (grid.isEmpty()) return 0
        if (grid[0].isEmpty()) return 0
        val stack = ArrayDeque<IntArray>()
        var res = 0
        for ((i, array) in grid.withIndex()) {
            for ((j, n) in array.withIndex()) {
                if (n == 0) {
                    if (i == 0 || i == grid.size - 1 || j == 0 || j == array.size - 1) {
                        // 本身在最外层则所有相邻连续的0都不满足条件
                    } else {
                        // 判断是否满足条件
                        stack.clear()
                        stack.push(intArrayOf(i, j))
                        var isMatch = true
                        while (stack.isNotEmpty()) {
                            val temp = stack.pop()
                            if (temp[0] == 0 || temp[0] == grid.size - 1 || temp[1] == 0 || temp[1] == array.size - 1) {
                                isMatch = false
                            }

                            if (temp[0] > 0 && grid[temp[0] - 1][temp[1]] == 0) {
                                grid[temp[0] - 1][temp[1]] = -1
                                stack.push(intArrayOf(temp[0] - 1, temp[1]))
                            }

                            if (temp[0] < grid.size - 1 && grid[temp[0] + 1][temp[1]] == 0) {
                                grid[temp[0] + 1][temp[1]] = -1
                                stack.push(intArrayOf(temp[0] + 1, temp[1]))
                            }

                            if (temp[1] > 0 && grid[temp[0]][temp[1] - 1] == 0) {
                                grid[temp[0]][temp[1] - 1] = -1
                                stack.push(intArrayOf(temp[0], temp[1] - 1))
                            }

                            if (temp[1] < array.size - 1 && grid[temp[0]][temp[1] + 1] == 0) {
                                grid[temp[0]][temp[1] + 1] = -1
                                stack.push(intArrayOf(temp[0], temp[1] + 1))
                            }
                        }

                        if (isMatch) {
                            res++
                        }
                    }
                }
            }
        }
        return res
    }
}
```
### 52
```
Construct Quad Tree

Given a n * n matrix grid of 0's and 1's only. We want to represent the grid with a Quad-Tree.

Return the root of the Quad-Tree representing the grid.

Notice that you can assign the value of a node to True or False when isLeaf is False, and both are accepted in the answer.

A Quad-Tree is a tree data structure in which each internal node has exactly four children. Besides, each node has two attributes:
val: True if the node represents a grid of 1's or False if the node represents a grid of 0's. 
isLeaf: True if the node is leaf node on the tree or False if the node has the four children.
class Node {
    public boolean val;
    public boolean isLeaf;
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;
}

We can construct a Quad-Tree from a two-dimensional area using the following steps:
If the current grid has the same value (i.e all 1's or all 0's) set isLeaf True and set val to the value of the grid and set the four children to Null and stop.
If the current grid has different values, set isLeaf to False and set val to any value and divide the current grid into four sub-grids as shown in the photo.
Recurse for each of the children with the proper sub-grid.

If you want to know more about the Quad-Tree, you can refer to the wiki.

Quad-Tree format:
The output represents the serialized format of a Quad-Tree using level order traversal, where null signifies a path terminator where no node exists below.
It is very similar to the serialization of the binary tree. The only difference is that the node is represented as a list [isLeaf, val].
If the value of isLeaf or val is True we represent it as 1 in the list [isLeaf, val] and if the value of isLeaf or val is False we represent it as 0.

Example 1:
Input: grid = [[0,1],[1,0]]
Output: [[0,1],[1,0],[1,1],[1,1],[1,0]]
Explanation: The explanation of this example is shown below:
Notice that 0 represnts False and 1 represents True in the photo representing the Quad-Tree.

Example 2:
Input: grid = [[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]]
Output: [[0,1],[1,1],[0,1],[1,1],[1,0],null,null,null,null,[1,0],[1,0],[1,1],[1,1]]
Explanation: All values in the grid are not the same. We divide the grid into four sub-grids.
The topLeft, bottomLeft and bottomRight each has the same value.
The topRight have different values so we divide it into 4 sub-grids where each has the same value.
Explanation is shown in the photo below:

Example 3:
Input: grid = [[1,1],[1,1]]
Output: [[1,1]]

Example 4:
Input: grid = [[0]]
Output: [[1,0]]

Example 5:
Input: grid = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
Output: [[0,1],[1,1],[1,0],[1,0],[1,1]]

[
[1,1,1,1,0,0,0,0],
[1,1,1,1,0,0,0,0],
[1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1],
[1,1,1,1,0,0,0,0],
[1,1,1,1,0,0,0,0],
[1,1,1,1,0,0,0,0],
[1,1,1,1,0,0,0,0]]

Constraints:
n == grid.length == grid[i].length
n == 2^x where 0 <= x <= 6
```
##### mine
```
/**
 * Definition for a QuadTree node.
 * class Node(var `val`: Boolean, var isLeaf: Boolean) {
 *     var topLeft: Node? = null
 *     var topRight: Node? = null
 *     var bottomLeft: Node? = null
 *     var bottomRight: Node? = null
 * }
 */

class Solution {
    fun construct(grid: Array<IntArray>): Node? {
        return construct1(grid, 0, 0, grid.size, grid.size)
    }
    
    private fun construct1(
        grid: Array<IntArray>,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): Node? {
        if (grid.isEmpty()) return null
        if (right - left == 1) {
            return Node(grid[top][left] == 1, true)
        }

        if (right - left > 1) {
            val tl =
                construct1(grid, left, top, (right - left) / 2 + left, (bottom - top) / 2 + top)
            val tr =
                construct1(grid, (right - left) / 2 + left, top, right, (bottom - top) / 2 + top)
            val bl =
                construct1(grid, left, (bottom - top) / 2 + top, (right - left) / 2 + left, bottom)
            val br =
                construct1(grid, (right - left) / 2 + left, (bottom - top) / 2 + top, right, bottom)
            return if (
                (tl?.`val` == tr?.`val` && bl?.`val` == br?.`val` && tl?.`val` == bl?.`val`)
                && (tl?.isLeaf == tr?.isLeaf && bl?.isLeaf == br?.isLeaf && tl?.isLeaf == bl?.isLeaf && tl?.isLeaf == true)
            ) {
                Node(tl.`val`, true)
            } else {
                Node(true, false).apply {
                    topLeft = tl
                    topRight = tr
                    bottomLeft = bl
                    bottomRight = br
                }
            }
        }

        return null
    }
}
```
### 53
```
Sqrt(x)

Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:
Input: 4
Output: 2

Example 2:
Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
```
##### mine
```
class Solution {
    fun mySqrt(x: Int): Int {
        if (x == 0) return 0
        var left = 1
        var right = x
        while (left <= right) {
            val mid = left + (right - left) / 2
            when {
                mid == x / mid -> {
                    return mid
                }
                mid < x / mid -> {
                    left = mid + 1
                }
                else -> {
                    right = mid - 1
                }
            }
        }
        return right
    }
}
```
### 54
```
Number of Dice Rolls With Target Sum

You have d dice, and each die has f faces numbered 1, 2, ..., f.

Return the number of possible ways (out of fd total ways) modulo 10^9 + 7 to roll the dice so the sum of the face up numbers equals target.

Example 1:
Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.

Example 2:
Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.

Example 3:
Input: d = 2, f = 5, target = 10
Output: 1
Explanation: 
You throw two dice, each with 5 faces.  There is only one way to get a sum of 10: 5+5.

Example 4:
Input: d = 1, f = 2, target = 3
Output: 0
Explanation: 
You throw one die with 2 faces.  There is no way to get a sum of 3.

Example 5:
Input: d = 30, f = 30, target = 500
Output: 222616187
Explanation: 
The answer must be returned modulo 10^9 + 7.

Constraints:
1 <= d, f <= 30
1 <= target <= 1000
```
##### mine Time Limit Exceeded
```
fun numRollsToTarget(d: Int, f: Int, target: Int): Int {
        var res = 0
        if (target > f) {
            if (d > 1) {
                for (i in 1..f) {
                    res += numRollsToTarget(d - 1, f, target - i)
                }
            }
        } else {
            if (d > 1) {
                for (i in 1 until target) {
                    res += numRollsToTarget(d - 1, f, target - i)
                }
            } else {
                res += 1
            }
        }
        return res
    }

```
##### like
```
class Solution {
    fun numRollsToTarget(d: Int, f: Int, target: Int): Int {
        if (d == 0) {
            return if (0 == target) 1 else 0
        }
        val mod = 1000000007
        val array = Array(d + 1) {
            IntArray(target + 1) {
                0
            }
        }
        array[0][0] = 1
        for (i in 1..d) {
            for (j in 1..target) {
                if (j > i * f) continue else {
                    var k = 1
                    while (k <= f && k <= j){
                        array[i][j] = (array[i][j] + array[i - 1][j-k]) % mod
                        k++
                    }
                }
            }
        }
        return array[d][target]
    }
}
```
### 55
```
Minimum Size Subarray Sum

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

Example: 

Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.

Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n). 
```
##### mine
```
class Solution {
    fun minSubArrayLen(s: Int, nums: IntArray): Int {
        val queue = ArrayDeque<Int>()
        var res = Int.MAX_VALUE
        var sum = 0
        for (i in nums) {
            sum += i
            queue.push(i)
            while (sum >= s) {
                res = Math.min(res, queue.size)
                sum -= queue.removeLast()
            }
        }
        return if(res == Int.MAX_VALUE) 0 else res
    }
}
```
### 56
```
Construct String from Binary Tree

You need to construct a string consists of parenthesis and integers from a binary tree with the preorder traversing way.

The null node needs to be represented by empty parenthesis pair "()". And you need to omit all the empty parenthesis pairs that don't affect the one-to-one mapping relationship between the string and the original binary tree.

Example 1:
Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     

Output: "1(2(4))(3)"
Explanation: Originallay it needs to be "1(2(4)())(3()())", 
but you need to omit all the unnecessary empty parenthesis pairs. 
And it will be "1(2(4))(3)".

Example 2:
Input: Binary tree: [1,2,3,null,4]
       1
     /   \
    2     3
     \  
      4 

Output: "1(2()(4))(3)"
Explanation: Almost the same as the first example, 
except we can't omit the first parenthesis pair to break the one-to-one mapping relationship between the input and the output.
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
    fun tree2str(t: TreeNode?): String {
        if (t == null) return ""
        var res = "${t.`val`}"
        if (t.left != null) {
            res = "$res(${tree2str(t.left)})"
        } else {
            if (t.right != null) {
                res = "$res()"
            }
        }

        if (t.right != null) {
            res = "$res(${tree2str(t.right)})"
        }
        return res
    }
}
```
### 57
```
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Example 1:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

Example 2:
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]

Example 3:
Input: candidates = [2], target = 1
Output: []

Example 4:
Input: candidates = [1], target = 1
Output: [[1]]

Example 5:
Input: candidates = [1], target = 2
Output: [[1,1]]

Constraints:
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
All elements of candidates are distinct.
1 <= target <= 500
```
##### mine
```
class Solution {
    fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        val temp = mutableListOf<Int>()
        val list = candidates.filter {
            it <= target
        }.sortedBy {
            it
        }
        combinationSumSub(list, target, res, temp)
        return res
    }

    fun combinationSumSub(
        candidates: List<Int>,
        target: Int,
        res: MutableList<List<Int>>,
        temp: MutableList<Int>
    ) {
        when {
            target == 0 -> {
                res.add(mutableListOf<Int>().apply {
                    addAll(temp)
                })
                return
            }
            target < 0 -> {
                return
            }
            else -> {
                for (num in candidates) {
                    if (temp.isNotEmpty() && temp[temp.size - 1] > num) {
                        continue
                    }
                    temp.add(num)
                    combinationSumSub(candidates, target - num, res, temp)
                    temp.removeAt(temp.size - 1)
                }
            }
        }
    }
}
```
### 58
```
Zuma Game

Think about Zuma Game. You have a row of balls on the table, colored red(R), yellow(Y), blue(B), green(G), and white(W). You also have several balls in your hand.

Each time, you may choose a ball in your hand, and insert it into the row (including the leftmost place and rightmost place). Then, if there is a group of 3 or more balls in the same color touching, remove these balls. Keep doing this until no more balls can be removed.

Find the minimal balls you have to insert to remove all the balls on the table. If you cannot remove all the balls, output -1.

Example 1:
Input: board = "WRRBBW", hand = "RB"
Output: -1
Explanation: WRRBBW -> WRR[R]BBW -> WBBW -> WBB[B]W -> WW

Example 2:
Input: board = "WWRRBBWW", hand = "WRBRW"
Output: 2
Explanation: WWRRBBWW -> WWRR[R]BBWW -> WWBBWW -> WWBB[B]WW -> WWWW -> empty

Example 3:
Input: board = "G", hand = "GGGGG"
Output: 2
Explanation: G -> G[G] -> GG[G] -> empty 

Example 4:
Input: board = "RBYYBBRRB", hand = "YRBGB"
Output: 3
Explanation: RBYYBBRRB -> RBYY[Y]BBRRB -> RBBBRRB -> RRRB -> B -> B[B] -> BB[B] -> empty 

Constraints:
You may assume that the initial row of balls on the table won’t have any 3 or more consecutive balls with the same color.
1 <= board.length <= 16
1 <= hand.length <= 5
Both input strings will be non-empty and only contain characters 'R','Y','B','G','W'.
```
##### like
```
class Solution {
    fun findMinStep(board: String, hand: String): Int {
        val handCount = IntArray(26) {
            0
        }
        for (i in hand.indices) handCount[hand[i] - 'A']++
        val rs = helper("$board#", handCount)
        return if (rs == 6) -1 else rs
    }

    fun helper(s: String, h: IntArray): Int {
        var ss = remove(s)
        if (ss == "#") return 0
        var rs = 6
        var need = 0
        var i = 0
        var j = 0
        while (j < ss.length) {
            if (ss[i] == ss[j]) {
                j++
                continue
            }
            need = 3 - (j - i)
            if (h[ss[i] - 'A'] >= need) {
                h[ss[i] - 'A'] -= need
                rs = Math.min(rs, need + helper(ss.substring(0, i) + ss.substring(j), h))
                h[ss[i] - 'A'] += need
            }
            i = j
        }
        return rs
    }

    fun remove(board: String): String {
        var i = 0
        var j = 0
        while (j < board.length) {
            if (board[i] == board[j]) {
                j++
                continue
            }
            if (j - i >= 3)
                return remove(board.substring(0, i) + board.substring(j))
            i = j
            j++
        }
        return board
    }
}
```
### 59
```
Knight Probability in Chessboard

On an NxN chessboard, a knight starts at the r-th row and c-th column and attempts to make exactly K moves. The rows and columns are 0 indexed, so the top-left square is (0, 0), and the bottom-right square is (N-1, N-1).

A chess knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

Each time the knight is to move, it chooses one of eight possible moves uniformly at random (even if the piece would go off the chessboard) and moves there.

The knight continues moving until it has made exactly K moves or has moved off the chessboard. Return the probability that the knight remains on the board after it has stopped moving.

Example:
Input: 3, 2, 0, 0
Output: 0.0625
Explanation: There are two moves (to (1,2), (2,1)) that will keep the knight on the board.
From each of those positions, there are also two moves that will keep the knight on the board.
The total probability the knight stays on the board is 0.0625.

Note:
N will be between 1 and 25.
K will be between 0 and 100.
The knight always initially starts on the board.
```
##### mine 1 Time Limit Exceeded
```
class Solution {
    val array = arrayOf(
        intArrayOf(1, 2),
        intArrayOf(1, -2),
        intArrayOf(2, 1),
        intArrayOf(2, -1),
        intArrayOf(-1, 2),
        intArrayOf(-1, -2),
        intArrayOf(-2, 1),
        intArrayOf(-2, -1)
    )

    fun knightProbability(N: Int, K: Int, r: Int, c: Int): Double {
        if (K == 0) {
            return if (isLegal(r, c, N)) 1.0 else 0.0
        }

        if (K == 1) {
            var legalCount = 0
            for (a in array) {
                if (isLegal(r + a[0], c + a[1], N)) {
                    legalCount++
                }
            }
            return legalCount / 8.0
        }
        var res = 0.0
        for (a in array) {
            if (isLegal(r + a[0], c + a[1], N)) {
                res += 0.125 * knightProbability(N, K - 1, r + a[0], c + a[1])
            }
        }
        return res
    }

    fun isLegal(r: Int, c: Int, len: Int): Boolean = r in 0 until len && (c in 0 until len)
}
```
##### like
```
class Solution {
    val array = arrayOf(
        intArrayOf(1, 2),
        intArrayOf(1, -2),
        intArrayOf(2, 1),
        intArrayOf(2, -1),
        intArrayOf(-1, 2),
        intArrayOf(-1, -2),
        intArrayOf(-2, 1),
        intArrayOf(-2, -1)
    )
    
    private lateinit var dp: Array<Array<DoubleArray>>
    
    fun knightProbability(N: Int, K: Int, r: Int, c: Int): Double {
        dp = Array(N) {
            Array(N) {
                DoubleArray(
                    K + 1
                )
            }
        }
        return find(N, K, r, c)
    }

    fun find(N: Int, K: Int, r: Int, c: Int): Double {
        if (r < 0 || r > N - 1 || c < 0 || c > N - 1) return 0.0
        if (K == 0) return 1.0
        if (dp[r][c][K] != 0.0) return dp[r][c][K]
        var rate = 0.0
        for (i in array.indices) rate += 0.125 * find(N, K - 1, r + array[i][0], c + array[i][1])
        dp[r][c][K] = rate
        return rate
    }
}
```
### 60
```
Convert Binary Number in a Linked List to Integer

Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

Example 1:
Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10

Example 2:
Input: head = [0]
Output: 0

Example 3:
Input: head = [1]
Output: 1

Example 4:
Input: head = [1,0,0,1,0,0,1,1,1,0,0,0,0,0,0]
Output: 18880

Example 5:
Input: head = [0,0]
Output: 0
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
    fun getDecimalValue(head: ListNode?): Int {
        var a = reverse(head)
        var res = 0
        var mode = 0.0
        while (a != null) {
            res += a.`val` * Math.pow(2.0, mode).toInt()
            mode += 1.0
            a = a.next
        }
        return res
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
##### like
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
    fun getDecimalValue(head: ListNode?): Int {
        var res = 0
        var a = head
        while (a != null) {
            res = (res.shl(1).or(a.`val`))
            a = a.next
        }
        return res
    }
}
```
### 61
```
Given a chemical formula (given as a string), return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow. For example, H2O and H2O2 are possible, but H1O2 is impossible.

Two formulas concatenated together to produce another formula. For example, H2O2He3Mg4 is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula. For example, (H2O2) and (H2O2)3 are formulas.

Given a formula, return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.

Example 1:
Input: formula = "H2O"
Output: "H2O"
Explanation: The count of elements are {'H': 2, 'O': 1}.

Example 2:
Input: formula = "Mg(OH)2"
Output: "H2MgO2"
Explanation: The count of elements are {'H': 2, 'Mg': 1, 'O': 2}.

Example 3:
Input: formula = "K4(ON(SO3)2)2"
Output: "K4N2O14S4"
Explanation: The count of elements are {'K': 4, 'N': 2, 'O': 14, 'S': 4}.

Example 4:
Input: formula = "Be32"
Output: "Be32"

Constraints:
1 <= formula.length <= 1000
formula consists of English letters, digits, '(', and ')'.
formula is always valid.
```
### mine
```
class Solution {
    fun countOfAtoms(formula: String): String {
        val resMap = countOfAtomsSub(0, formula.length - 1, formula).toSortedMap()
        var res = ""
        resMap.forEach { (t, u) ->
            res += t
            if(u != 1) {
                res += u
            }
        }
        return res
    }

    fun countOfAtomsSub(startIndex: Int, endIndex: Int, formula: String): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        var index = startIndex
        var curLetter = ""
        while (index <= endIndex) {
            when {
                formula[index] == '(' -> {
                    if (curLetter.isNotEmpty()) {
                        if (map.containsKey(curLetter)) {
                            map[curLetter] = map[curLetter]!! + 1
                        } else {
                            map[curLetter] = 1
                        }
                        curLetter = ""
                    }
                    val adapterIndex = getParenthesesAdapterIndex(index, formula)
                    val childMul = getChildMul(adapterIndex, formula)
                    val childMap = countOfAtomsSub(index + 1, adapterIndex - 1, formula)
                    childMap.forEach { (t, u) ->
                        if (map.containsKey(t)) {
                            map[t] = map[t]!! + u * childMul.first
                        } else {
                            map[t] = u * childMul.first
                        }
                    }
                    index = adapterIndex + childMul.second + 1
                }
                formula[index] == ')' -> {
                    if (curLetter.isNotEmpty()) {
                        if (map.containsKey(curLetter)) {
                            map[curLetter] = map[curLetter]!! + 1
                        } else {
                            map[curLetter] = 1
                        }
                        curLetter = ""
                    }
                    index++
                }
                formula[index] in 'A'..'Z' -> {
                    if (curLetter.isEmpty()) {
                        curLetter += formula[index]
                    } else {
                        if (map.containsKey(curLetter)) {
                            map[curLetter] = map[curLetter]!! + 1
                        } else {
                            map[curLetter] = 1
                        }
                        curLetter = "" + formula[index]
                    }
                    index++
                }
                formula[index] in 'a'..'z' -> {
                    curLetter += formula[index]
                    index++
                }
                formula[index] in '0'..'9' -> {
                    val count = getChildMul(index - 1, formula)
                    if (map.containsKey(curLetter)) {
                        map[curLetter] = map[curLetter]!! + count.first
                    } else {
                        map[curLetter] = count.first
                    }
                    curLetter = ""
                    index += count.second
                }
            }
        }
        if(curLetter.isNotEmpty()) {
            if (map.containsKey(curLetter)) {
                map[curLetter] = map[curLetter]!! + 1
            } else {
                map[curLetter] = 1
            }
        }
        return map
    }

    fun getChildMul(index: Int, formula: String): Pair<Int, Int> {
        if (index == formula.length - 1) return Pair(1, 0)
        var i = index + 1
        var res = ""
        while (i < formula.length) {
            if (formula[i] in '0'..'9')
                res += formula[i]
            else
                break
            i++
        }
        return if (res.isEmpty()) Pair(1, 0) else Pair(res.toInt(), res.length)
    }

    fun getParenthesesAdapterIndex(index: Int, formula: String): Int {
        var parenthesesCount = 0
        var i = index
        while (i < formula.length) {
            if (formula[i] == '(')
                parenthesesCount++
            else if (formula[i] == ')') {
                parenthesesCount--
            }
            if (parenthesesCount == 0) {
                return i
            }
            i++
        }
        throw IllegalStateException("getParenthesesAdapterIndex fail")
    }
}
```
### 62
```
Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.

Constraints:
intervals[i][0] <= intervals[i][1]
```
### mine
```
class Solution {
    fun merge(intervals: Array<IntArray>): Array<IntArray> {
        if(intervals.isNullOrEmpty()) return arrayOf()
        val list = mutableListOf<IntArray>().apply {
            add(intervals[0])
        }
        var index = 1
        while (index < intervals.size) {
            mergeChild(intervals[index], list)
            index++
        }
        return list.toTypedArray()
    }

    fun mergeChild(array: IntArray, list: MutableList<IntArray>) {
        var merge = false
        val iter = list.iterator()
        while (iter.hasNext()) {
            val cur = iter.next()
            if (array[0] >= cur[0] && array[1] <= cur[1]) {
                merge = true
            } else if (array[0] <= cur[0] && array[1] >= cur[1]) {
                iter.remove()
            } else if (array[0] in cur[0]..cur[1]) {
                array[0] = cur[0]
                iter.remove()
            } else if (array[1] in cur[0]..cur[1]) {
                array[1] = cur[1]
                iter.remove()
            }
        }
        if (!merge) {
            list.add(array)
        }
    }
}
```
### 63
```
Add to Array-Form of Integer

For a non-negative integer X, the array-form of X is an array of its digits in left to right order.  For example, if X = 1231, then the array form is [1,2,3,1].

Given the array-form A of a non-negative integer X, return the array-form of the integer X+K.

Example 1:
Input: A = [1,2,0,0], K = 34
Output: [1,2,3,4]
Explanation: 1200 + 34 = 1234

Example 2:
Input: A = [2,7,4], K = 181
Output: [4,5,5]
Explanation: 274 + 181 = 455

Example 3:
Input: A = [2,1,5], K = 806
Output: [1,0,2,1]
Explanation: 215 + 806 = 1021

Example 4:
Input: A = [9,9,9,9,9,9,9,9,9,9], K = 1
Output: [1,0,0,0,0,0,0,0,0,0,0]
Explanation: 9999999999 + 1 = 10000000000

Note：
1 <= A.length <= 10000
0 <= A[i] <= 9
0 <= K <= 10000
If A.length > 1, then A[0] != 0
```
##### mine 1 very slow
```
class Solution {
    fun addToArrayForm(A: IntArray, K: Int): List<Int> {
        val kk = K.toString().map {
            it - '0'
        }.toMutableList()
        var tmp = 0
        var ki = kk.size - 1
        var i = A.size - 1
        var div = 10
        while (i >= 0 && ki >= 0) {
            val value = A[i] + kk[ki] + tmp
            tmp = value / div
            kk[ki] = value % div
            i--
            ki--
        }
        while (i >= 0) {
            val value = A[i] + tmp
            tmp = value / div
            kk.add(0, value % div)
            i--
        }
        while (ki >= 0) {
            val value = kk[ki] + tmp
            tmp = value / div
            kk[ki] = value % div
            ki--
        }
        if (tmp > 0) {
            kk.add(0, tmp)
        }
        return kk
    }
}
```
##### mine
```
class Solution {
    fun addToArrayForm(A: IntArray, K: Int): List<Int> {
        val res = mutableListOf<Int>()
        val kk = K.toString()
        var tmp = 0
        var ki = kk.length - 1
        var i = A.size - 1
        val div = 10
        while (i >= 0 && ki >= 0) {
            val value = A[i] + (kk[ki] - '0') + tmp
            tmp = value / div
            res.add(0, value % div)
            i--
            ki--
        }
        while (i >= 0) {
            val value = A[i] + tmp
            tmp = value / div
            res.add(0, value % div)
            i--
        }
        while (ki >= 0) {
            val value = (kk[ki] - '0') + tmp
            tmp = value / div
            res.add(0, value % div)
            ki--
        }
        if (tmp > 0) {
            res.add(0, tmp)
        }
        return res
    }
}
```
##### like
```
class Solution {
    fun addToArrayForm(A: IntArray, K: Int): List<Int> {
        val res = mutableListOf<Int>()
        var KK = K
        for (i in A.size - 1 downTo 0) {
            res.add(0, (A[i] + KK) % 10)
            KK = (A[i] + KK) / 10
        }
        while (KK > 0) {
            res.add(0, KK % 10)
            KK /= 10
        }
        return res
    }
}
```
### 64
```
Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements. 

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows
a, b are from arr
a < b
b - a equals to the minimum absolute difference of any two elements in arr

Example 1:
Input: arr = [4,2,1,3]
Output: [[1,2],[2,3],[3,4]]
Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.

Example 2:
Input: arr = [1,3,6,10,15]
Output: [[1,3]]

Example 3:
Input: arr = [3,8,-10,23,19,-4,-14,27]
Output: [[-14,-10],[19,23],[23,27]]

Constraints:
2 <= arr.length <= 10^5
-10^6 <= arr[i] <= 10^6
```
##### mine
```
class Solution {
    fun minimumAbsDifference(arr: IntArray): List<List<Int>> {
        arr.sort()
        val map = mutableMapOf<Int, MutableList<List<Int>>>()
        var i = 1
        while (i < arr.size) {
            val key = arr[i] - arr[i - 1]
            if (map.containsKey(key)) {
                map[key]!!.add(listOf(arr[i - 1], arr[i]))
            } else {
                map[key] = mutableListOf(listOf(arr[i - 1], arr[i]))
            }
            i++
        }

        return map[map.keys.min()!!]!!.toList()
    }
}
```
### 65
```
Get Watched Videos by Your Friends

There are n people, each person has a unique id between 0 and n-1. Given the arrays watchedVideos and friends, where watchedVideos[i] and friends[i] contain the list of watched videos and the list of friends respectively for the person with id = i.

Level 1 of videos are all watched videos by your friends, level 2 of videos are all watched videos by the friends of your friends and so on. In general, the level k of videos are all watched videos by people with the shortest path exactly equal to k with you. Given your id and the level of videos, return the list of videos ordered by their frequencies (increasing). For videos with the same frequency order them alphabetically from least to greatest. 

Example 1:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 1
Output: ["B","C"] 
Explanation: 
You have id = 0 (green color in the figure) and your friends are (yellow color in the figure):
Person with id = 1 -> watchedVideos = ["C"] 
Person with id = 2 -> watchedVideos = ["B","C"] 
The frequencies of watchedVideos by your friends are: 
B -> 1 
C -> 2

Example 2:
Input: watchedVideos = [["A","B"],["C"],["B","C"],["D"]], friends = [[1,2],[0,3],[0,3],[1,2]], id = 0, level = 2
Output: ["D"]
Explanation: 
You have id = 0 (green color in the figure) and the only friend of your friends is the person with id = 3 (yellow color in the figure).

Constraints:
n == watchedVideos.length == friends.length
2 <= n <= 100
1 <= watchedVideos[i].length <= 100
1 <= watchedVideos[i][j].length <= 8
0 <= friends[i].length < n
0 <= friends[i][j] < n
0 <= id < n
1 <= level < n
if friends[i] contains j, then friends[j] contains i
```
##### mine
```
class Solution {
    fun watchedVideosByFriends(watchedVideos: List<List<String>>, friends: Array<IntArray>, id: Int, level: Int): List<String> {
        var tlevel = 0
        var curSize = 0
        val ids = mutableListOf<Int>().apply {
            add(id)
        }
        val oids = mutableSetOf<Int>()
        while (ids.isNotEmpty()) {
            if(tlevel == level) break
            if(curSize == 0) {
                curSize = ids.size
            }
            val curId = ids.removeAt(0)
            if(!oids.contains(curId)) {
                friends[curId].forEach {
                    if (!oids.contains(it)) {
                        ids.add(it)
                    }
                }
            }
            oids.add(curId)
            curSize--
            if(curSize == 0) {
                tlevel++
            }
        }

        val map = mutableMapOf<String, Int>()
        val res = mutableListOf<String>()
        ids.forEach {
            watchedVideos[it].forEach { vs ->
                if (map.containsKey(vs)) {
                    map[vs] = map[vs]!! + 1
                } else {
                    map[vs] = 1
                }
            }
        }
        map.keys.forEach {
            res.add(it)
        }
        res.sortWith(Comparator { o1: String, o2: String ->
            if (map[o1] == map[o2]) o1.compareTo(o2) else map[o1]!! - map[o2]!!
        })
        return res
    }
}
```
### 66
```
Classes More Than 5 Students

There is a table courses with columns: student and class

Please list out all classes which have more than or equal to 5 students.

For example, the table:
+---------+------------+
| student | class      |
+---------+------------+
| A       | Math       |
| B       | English    |
| C       | Math       |
| D       | Biology    |
| E       | Math       |
| F       | Computer   |
| G       | Math       |
| H       | Math       |
| I       | Math       |
+---------+------------+

Should output:
+---------+
| class   |
+---------+
| Math    |
+---------+

Note:
The students should not be counted duplicate in each course.
```
##### answer
```
select class from courses group by class having count(distinct student) >= 5;
```
### 67
```
Split Linked List in Parts

Given a (singly) linked list with head node root, write a function to split the linked list into k consecutive linked list "parts".

The length of each part should be as equal as possible: no two parts should have a size differing by more than 1. This may lead to some parts being null.

The parts should be in order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal parts occurring later.

Return a List of ListNode's representing the linked list parts that are formed.

Examples 1->2->3->4, k = 5 // 5 equal parts [ [1], [2], [3], [4], null ]

Example 1:
Input:
root = [1, 2, 3], k = 5
Output: [[1],[2],[3],[],[]]
Explanation:
The input and each element of the output are ListNodes, not arrays.
For example, the input root has root.val = 1, root.next.val = 2, \root.next.next.val = 3, and root.next.next.next = null.
The first element output[0] has output[0].val = 1, output[0].next = null.
The last element output[4] is null, but it's string representation as a ListNode is [].

Example 2:
Input: 
root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
Explanation:
The input has been split into consecutive parts with size difference at most 1, and earlier parts are a larger size than the later parts.

Note:
The length of root will be in the range [0, 1000].
Each value of a node in the input will be an integer in the range [0, 999].
k will be an integer in the range [1, 50].
```
### mine 100% 100%
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
    fun splitListToParts(root: ListNode?, k: Int): Array<ListNode?> {
        val res = Array<ListNode?>(k) { null }
        if (root == null) return res
        val link = mutableListOf<ListNode>()
        var cur = root
        while (cur != null) {
            link.add(cur)
            cur = cur.next
        }
        if (link.size <= k) {
            var i = 0
            while (i < link.size) {
                res[i] = link[i].apply {
                    next = null
                }
                i++
            }
            return res
        }

        val small = link.size / k
        val big = small + 1
        var useSmall = true
        val smallMaxCount = (link.size - (link.size - small * k) * big) / small
        var smallCount = 0
        var kk = res.size
        var index = link.size
        while (index > 0) {
            if (smallMaxCount == smallCount) useSmall = false
            if (index != link.size) {
                link[index - 1].next = null
            }
            if (useSmall) {
                res[kk - 1] = link[index - small]
                index -= small
                smallCount++
            } else {
                res[kk - 1] = link[index - big]
                index -= big
            }
            kk--
        }
        return res
    }
}
```
### 68
```
Element Appearing More Than 25% In Sorted Array

Given an integer array sorted in non-decreasing order, there is exactly one integer in the array that occurs more than 25% of the time.

Return that integer.

Example 1:
Input: arr = [1,2,2,6,6,6,6,7,10]
Output: 6

Constraints:
1 <= arr.length <= 10^4
0 <= arr[i] <= 10^5
```
##### mine
```
class Solution {
    fun findSpecialInteger(arr: IntArray): Int {
        val aliquot = arr.size % 4 == 0
        val percentCount = if (aliquot) arr.size / 4 else arr.size / 4 + 1
        var count = 1
        var cur = -1
        for (i in arr) {
            if (cur != i) {
                cur = i
                count = 1
            } else {
                count++
            }

            if (aliquot && count > percentCount) {
                return cur
            }

            if(!aliquot && count >= percentCount) {
                return cur
            }
        }
        return -1
    }
}
```
##### like
```
class Solution {
    fun findSpecialInteger(arr: IntArray): Int {
        val indexs = intArrayOf(arr.size / 4, arr.size / 2, arr.size / 4 * 3)
        for (index in indexs) {
            val firstIndex = findFirstIndex(arr, index)
            if (arr[firstIndex] == arr[firstIndex + arr.size / 4]) {
                return arr[firstIndex]
            }
        }
        return -1
    }

    fun findFirstIndex(arr: IntArray, index: Int): Int {
        if (index == 0) return index
        var i = index
        while (i > -1) {
            if (arr[i] == arr[index])
                i--
            else
                return i + 1
        }
        return 0
    }
}
```
### 69
```
Goat Latin

A sentence S is given, composed of words separated by spaces. Each word consists of lowercase and uppercase letters only.

We would like to convert the sentence to "Goat Latin" (a made-up language similar to Pig Latin.)

The rules of Goat Latin are as follows:

If a word begins with a vowel (a, e, i, o, or u), append "ma" to the end of the word.
For example, the word 'apple' becomes 'applema'.
 
If a word begins with a consonant (i.e. not a vowel), remove the first letter and append it to the end, then add "ma".
For example, the word "goat" becomes "oatgma".
 
Add one letter 'a' to the end of each word per its word index in the sentence, starting with 1.
For example, the first word gets "a" added to the end, the second word gets "aa" added to the end and so on.
Return the final sentence representing the conversion from S to Goat Latin. 

Example 1:
Input: "I speak Goat Latin"
Output: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"

Example 2:
Input: "The quick brown fox jumped over the lazy dog"
Output: "heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"

Notes:
S contains only uppercase, lowercase and spaces. Exactly one space between each word.
1 <= S.length <= 150.
```
##### mine
```
class Solution {
    fun toGoatLatin(S: String): String {
        val strings = S.split(" ")
        var res = ""
        var index = 0
        for (str in strings) {
            var last = ""
            var i = 0
            while (i < index + 1) {
                last += 'a'
                i++
            }
            if (index != 0) {
                res += " "
            }
            res += if (str.isVowel()) {
                str + "ma" + last
            } else {
                val first = str[0]
                str.removeRange(0, 1) + first + "ma" + last
            }
            index++
        }
        return res   
    }
    
    fun String.isVowel(): Boolean {
        return this[0] == 'a' || this[0] == 'e' || this[0] == 'i' || this[0] == 'o' || this[0] == 'u'
                || this[0] == 'A' || this[0] == 'E' || this[0] == 'I' || this[0] == 'O' || this[0] == 'U'
    }
}
```
### 70
```
Guess Number Higher or Lower II

We are playing the Guessing Game. The game will work as follows:
    I pick a number between 1 and n.
    You guess a number.
    If you guess the right number, you win the game.
    If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.
    Every time you guess a wrong number x, you will pay x dollars. If you run out of money, you lose the game.

Given a particular n, return the minimum amount of money you need to guarantee a win regardless of what number I pick.

Example 1:
Input: n = 10
Output: 16
Explanation: The winning strategy is as follows:
- The range is [1,10]. Guess 7.
    - If this is my number, your total is $0. Otherwise, you pay $7.
    - If my number is higher, the range is [8,10]. Guess 9.
        - If this is my number, your total is $7. Otherwise, you pay $9.
        - If my number is higher, it must be 10. Guess 10. Your total is $7 + $9 = $16.
        - If my number is lower, it must be 8. Guess 8. Your total is $7 + $9 = $16.
    - If my number is lower, the range is [1,6]. Guess 3.
        - If this is my number, your total is $7. Otherwise, you pay $3.
        - If my number is higher, the range is [4,6]. Guess 5.
            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $5.
            - If my number is higher, it must be 6. Guess 6. Your total is $7 + $3 + $5 = $15.
            - If my number is lower, it must be 4. Guess 4. Your total is $7 + $3 + $5 = $15.
        - If my number is lower, the range is [1,2]. Guess 1.
            - If this is my number, your total is $7 + $3 = $10. Otherwise, you pay $1.
            - If my number is higher, it must be 2. Guess 2. Your total is $7 + $3 + $1 = $11.
The worst case in all these scenarios is that you pay $16. Hence, you only need $16 to guarantee a win.

Example 2:
Input: n = 1
Output: 0
Explanation: There is only one possible number, so you can guess 1 and not have to pay anything.

Example 3:
Input: n = 2
Output: 1
Explanation: There are two possible numbers, 1 and 2.
- Guess 1.
    - If this is my number, your total is $0. Otherwise, you pay $1.
    - If my number is higher, it must be 2. Guess 2. Your total is $1.
The worst case is that you pay $1.

Constraints:
1 <= n <= 200
```
##### like
```
class Solution {
    fun getMoneyAmount(n: Int): Int {
        val arr = Array<IntArray>(n + 1) {
            IntArray(n+1) {
                0
            }
        }
        return helperGet(arr, 1, n)
    }

    fun helperGet(array: Array<IntArray>, start: Int, end: Int) :Int {
        if(start >= end) return 0
        if(array[start][end] != 0) return array[start][end]
        var res = Int.MAX_VALUE
        var i = start
        while (i <= end) {
            val tmp = i + Math.max(helperGet(array, start, i - 1), helperGet(array, i + 1, end))
            res = Math.min(res, tmp)
            i++
        }
        array[start][end] = res
        return res
    }
}
``` 
### 71
```
Count Vowels Permutation

Given an integer n, your task is to count how many strings of length n can be formed under the following rules:

Each character is a lower case vowel ('a', 'e', 'i', 'o', 'u')
Each vowel 'a' may only be followed by an 'e'.
Each vowel 'e' may only be followed by an 'a' or an 'i'.
Each vowel 'i' may not be followed by another 'i'.
Each vowel 'o' may only be followed by an 'i' or a 'u'.
Each vowel 'u' may only be followed by an 'a'.
Since the answer may be too large, return it modulo 10^9 + 7.

Example 1:
Input: n = 1
Output: 5
Explanation: All possible strings are: "a", "e", "i" , "o" and "u".

Example 2:
Input: n = 2
Output: 10
Explanation: All possible strings are: "ae", "ea", "ei", "ia", "ie", "io", "iu", "oi", "ou" and "ua".

Example 3: 
Input: n = 5
Output: 68

Constraints:
1 <= n <= 2 * 10^4
```
##### mine 1  Memory Limit Exceeded
```
class Solution {
    fun countVowelPermutation(n: Int): Int {
        val queue = ArrayDeque<Char>()
        queue.add('a')
        queue.add('e')
        queue.add('i')
        queue.add('o')
        queue.add('u')
        var size = 0
        var times = 0
        while (queue.isNotEmpty()) {
            if (size == 0) {
                size = queue.size
                times++
                if (times == n) {
                    return size % 1000000007
                }
            }
            when (queue.pop()) {
                'a' -> {
                    queue.add('e')
                }
                'e' -> {
                    queue.add('a')
                    queue.add('i')
                }
                'i' -> {
                    queue.add('a')
                    queue.add('e')
                    queue.add('o')
                    queue.add('u')
                }
                'o' -> {
                    queue.add('i')
                    queue.add('u')
                }
                'u' -> {
                    queue.add('a')
                }
            }
            size--
        }
        return -1 % 1000000007
    }
}
```
##### mine 2 100 100
```
class Solution {
    fun countVowelPermutation(n: Int): Int {
        var aCount = 1L
        var eCount = 1L
        var iCount = 1L
        var oCount = 1L
        var uCount = 1L

        var aCount1 = 0L
        var eCount1 = 0L
        var iCount1 = 0L
        var oCount1 = 0L
        var uCount1 = 0L

        var times = 0
        while (times < n) {
            times++
            if(times == n) {
                return ((if(n % 2 == 0) aCount1 + eCount1 + iCount1 + oCount1 + uCount1 else aCount + eCount + iCount + oCount + uCount) % 1000000007L).toInt()
            }
            if(times % 2 == 0) {
                aCount = 0L
                eCount = 0L
                iCount = 0L
                oCount = 0L
                uCount = 0L

                eCount += aCount1

                aCount += eCount1
                iCount += eCount1

                aCount += iCount1
                eCount += iCount1
                oCount += iCount1
                uCount += iCount1

                iCount += oCount1
                uCount += oCount1

                aCount += uCount1

                aCount %= 1000000007L
                eCount %= 1000000007L
                iCount %= 1000000007L
                oCount %= 1000000007L
                uCount %= 1000000007L
            } else {
                aCount1 = 0L
                eCount1 = 0L
                iCount1 = 0L
                oCount1 = 0L
                uCount1 = 0L

                eCount1 += aCount

                aCount1 += eCount
                iCount1 += eCount

                aCount1 += iCount
                eCount1 += iCount
                oCount1 += iCount
                uCount1 += iCount

                iCount1 += oCount
                uCount1 += oCount

                aCount1 += uCount

                aCount1 %= 1000000007L
                eCount1 %= 1000000007L
                iCount1 %= 1000000007L
                oCount1 %= 1000000007L
                uCount1 %= 1000000007L
            }
        }
        return (-1L % 1000000007L).toInt()
    }
}
```
### 72
```
Convert Sorted List to Binary Search Tree

Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example 1:
Input: head = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: One possible answer is [0,-3,9,-10,null,5], which represents the shown height balanced BST.

Example 2:
Input: head = []
Output: []

Example 3:
Input: head = [0]
Output: [0]

Example 4:
Input: head = [1,3]
Output: [3,1]

Constraints:
The number of nodes in head is in the range [0, 2 * 104].
-10^5 <= Node.val <= 10^5
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
   fun sortedListToBST(head: ListNode?): TreeNode? {
        if (head == null) return null

        val list = mutableListOf<ListNode>()
        var cur = head
        while (cur != null) {
            list.add(cur)
            cur = cur.next
        }
        return sortedListToBSTHelp(list, 0, list.size - 1)
    }

    fun sortedListToBSTHelp(list: List<ListNode>, start: Int, end: Int): TreeNode? {
        if (start > end) return null
        if(start == end) return TreeNode(list[start].`val`)
        val center = start + (end - start) / 2
        return TreeNode(list[center].`val`).apply {
            left = sortedListToBSTHelp(list, start, center - 1)
            right = sortedListToBSTHelp(list, center + 1, end)
        }
    }
}
```
### 73
```
Number of Equivalent Domino Pairs

Given a list of dominoes, dominoes[i] = [a, b] is equivalent to dominoes[j] = [c, d] if and only if either (a==c and b==d), or (a==d and b==c) - that is, one domino can be rotated to be equal to another domino.

Return the number of pairs (i, j) for which 0 <= i < j < dominoes.length, and dominoes[i] is equivalent to dominoes[j].

Example 1:
Input: dominoes = [[1,2],[2,1],[3,4],[5,6]]
Output: 1

Constraints:
1 <= dominoes.length <= 40000
1 <= dominoes[i][j] <= 9
```
##### mine slow
```
class Solution {
    fun numEquivDominoPairs(dominoes: Array<IntArray>): Int {
        var res = 0

        var i = 0
        while (i < dominoes.size - 1) {
            val sum = dominoes[i][0] + dominoes[i][1]
            var j = i + 1
            while (j < dominoes.size) {
                if (sum == dominoes[j][0] + dominoes[j][1] && (dominoes[j][0] == dominoes[i][0] || dominoes[j][0] == dominoes[i][1])) {
                    res++
                }
                j++
            }
            i++
        }

        return res
    }
}
```
##### like
```
class Solution {
    fun numEquivDominoPairs(dominoes: Array<IntArray>): Int {
       val count: MutableMap<Int, Int> = HashMap()
        var res = 0
        for (d in dominoes) {
            val k = Math.min(d[0], d[1]) * 10 + Math.max(d[0], d[1])
            count[k] = count.getOrDefault(k, 0) + 1
        }
        for (v in count.values) {
            res += v * (v - 1) / 2
        }
        return res
    }
}
```
### 74
```
Insertion Sort List

Sort a linked list using insertion sort.

A graphical example of insertion sort. The partial sorted list (black) initially contains only the first element in the list.
With each iteration one element (red) is removed from the input data and inserted in-place into the sorted list
 
Algorithm of Insertion Sort:

Insertion sort iterates, consuming one input element each repetition, and growing a sorted output list.
At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there.
It repeats until no input elements remain.

Example 1:
Input: 4->2->1->3
Output: 1->2->3->4

Example 2:
Input: -1->5->3->4->0
Output: -1->0->3->4->5
```
##### mine 1
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
    fun insertionSortList(head: ListNode?): ListNode? {
        if (head == null) return null

        var cur = head.next
        val res = ListNode(0).apply {
            next = head
        }
        head.next = null

        while (cur != null) {
            val next = cur.next

            var cur2 = res.next
            var pre2: ListNode? = res
            while (cur2 != null) {
                if (cur2.`val` > cur.`val`) {
                    cur.next = cur2
                    pre2!!.next = cur
                    break
                }
                cur2 = cur2.next
                pre2 = pre2!!.next
            }
            if (cur2 == null) {
                pre2!!.next = cur
                cur.next = null
            }

            cur = next
        }
        return res.next
    }
}
```
##### min2
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
    fun insertionSortList(head: ListNode?): ListNode? {
        if (head == null) return null

        val list = mutableListOf<ListNode>()
        var cur = head
        while (cur != null) {
            list.add(cur)
            cur = cur.next
        }
        list.sortBy {
            it.`val`
        }
        val res = ListNode(0).apply {
            next = list[0]
        }
        if (list.size > 1) {
            cur = res.next
            for (i in 1 until list.size) {
                list[i].next = null
                cur!!.next = list[i]
                cur = cur.next
            }
        }

        return res.next
    }
}
```
### 75
```
Number of Ways to Stay in the Same Place After Some Steps

You have a pointer at index 0 in an array of size arrLen. At each step, you can move 1 position to the left, 1 position to the right in the array or stay in the same place  (The pointer should not be placed outside the array at any time).

Given two integers steps and arrLen, return the number of ways such that your pointer still at index 0 after exactly steps steps.

Since the answer may be too large, return it modulo 10^9 + 7.

Example 1:
Input: steps = 3, arrLen = 2
Output: 4
Explanation: There are 4 differents ways to stay at index 0 after 3 steps.
Right, Left, Stay 1
Stay, Right, Left
Right, Stay, Left 1
Stay, Stay, Stay 1

Example 2:
Input: steps = 2, arrLen = 4
Output: 2
Explanation: There are 2 differents ways to stay at index 0 after 2 steps
Right, Left
Stay, Stay

Example 3:
Input: steps = 4, arrLen = 2
Output: 8

Constraints:
1 <= steps <= 500
1 <= arrLen <= 10^6
```
##### mine
```
class Solution {
    fun numWays(steps: Int, arrLen: Int): Int {
        val len = if (steps / 2 + 1 > arrLen) arrLen else steps / 2 + 1
        val a = Array(steps + 1) {
            LongArray(len + 1) {
                0L
            }
        }
        a[1][0] = 1L
        a[1][1] = 1L
        for (i in 2 .. steps) {
            for (j in 0 until len) {
                if (a[i][j] == 0L) {
                    a[i][j] =
                        (a[i - 1][j] + (if (j < len - 1) a[i - 1][j + 1] else 0) + (if (j > 0) a[i - 1][j - 1] else 0)) % 1000000007L
                }
            }
        }
        return a[steps][0].toInt()
    }
}
```
### 76
```
Brace Expansion II

Under a grammar given below, strings can represent a set of lowercase words.  Let's use R(expr) to denote the set of words the expression represents.

Grammar can best be understood through simple examples:
    Single letters represent a singleton set containing that word.
        R("a") = {"a"}
        R("w") = {"w"}
    When we take a comma delimited list of 2 or more expressions, we take the union of possibilities.
        R("{a,b,c}") = {"a","b","c"}
        R("{{a,b},{b,c}}") = {"a","b","c"} (notice the final set only contains each word at most once)
    When we concatenate two expressions, we take the set of possible concatenations between two words where the first word comes from the first expression and the second word comes from the second expression.
        R("{a,b}{c,d}") = {"ac","ad","bc","bd"}
        R("a{b,c}{d,e}f{g,h}") = {"abdfg", "abdfh", "abefg", "abefh", "acdfg", "acdfh", "acefg", "acefh"}

Formally, the 3 rules for our grammar:
    For every lowercase letter x, we have R(x) = {x}
    For expressions e_1, e_2, ... , e_k with k >= 2, we have R({e_1,e_2,...}) = R(e_1) ∪ R(e_2) ∪ ...
    For expressions e_1 and e_2, we have R(e_1 + e_2) = {a + b for (a, b) in R(e_1) × R(e_2)}, where + denotes concatenation, and × denotes the cartesian product.
    Given an expression representing a set of words under the given grammar, return the sorted list of words that the expression represents.

Example 1:
Input: "{a,b}{c,{d,e}}"
Output: ["ac","ad","ae","bc","bd","be"]

Example 2:
Input: "{{a,z},a{b,c},{ab,z}}"
Output: ["a","ab","ac","z"]
Explanation: Each distinct word is written only once in the final answer.
 
Constraints:
    1 <= expression.length <= 60
    expression[i] consists of '{', '}', ','or lowercase English letters.
    The given expression represents a set of words based on the grammar given in the description.
```
##### mine too ugly
```
class Solution {
    fun braceExpansionII(expression: String): List<String> {
        val exp = "{${expression}}"
        val stack = ArrayDeque<String>()
        val set = mutableSetOf<String>()
        var i = 0
        while (i < exp.length) {
            val cur = exp[i]
            when (cur) {
                '{' -> {
                    stack.push(cur.toString())
                }
                '}' -> {
                    set.clear()
                    while (stack.isNotEmpty()) {
                        val pop = stack.pop()
                        if (pop == "{") {
                            break
                        }
                        if (pop != "," && pop != "[" && pop != "]") {
                            set.add(pop)
                        }
                    }
                    set.remove("")

                    var curValue = ""
                    var ii = i + 1
                    var directAdd = false
                    while (ii < exp.length) {
                        val next = exp[ii]
                        if (next in 'a'..'z') {
                            directAdd = true
                            curValue += next
                        } else {
                            break
                        }
                        ii++
                    }
                    if (directAdd) {
                        i = ii - 1
                    }
                    val isMul = i + 1 < exp.length && (exp[i + 1] in 'a'..'z' || exp[i + 1] == '{')
                    if (curValue.isNotEmpty()) {
                        val t = mutableSetOf<String>()
                        t.addAll(
                            set.map {
                                it + curValue
                            }
                        )
                        set.clear()
                        set.addAll(t)
                    }

                    if (stack.isEmpty()) {
                        stack.push("[")
                        set.forEach {
                            stack.push(it)
                        }
                        stack.push("]")
                    } else {
                        val pop = stack.pop()
                        if (pop == ",") {
                            if(isMul) {
                                stack.push(pop)
                                stack.push("[")
                                set.forEach {
                                    stack.push(it)
                                }
                                stack.push("]")
                            } else {
                                while (stack.isNotEmpty()) {
                                    val p = stack.pop()
                                    if (p == "{") {
                                        break
                                    }
                                    if (p == "[") {
                                        break
                                    }
                                    if (p != "," && p != "]") {
                                        set.add(p)
                                    }
                                }
                                set.remove("")

                                stack.push("[")
                                set.forEach {
                                    stack.push(it)
                                }
                                stack.push("]")
                            }
                        } else if (pop == "{") {
                            stack.push(pop)
                            stack.push("[")
                            set.forEach {
                                stack.push(it)
                            }
                            stack.push("]")
                        } else if (pop == "]") {
                            val t = mutableSetOf<String>()
                            while (stack.isNotEmpty()) {
                                val p = stack.pop()
                                if (p == "[") {
                                    break
                                }
                                t.addAll(
                                    set.map {
                                        p + it
                                    }
                                )
                            }
                            stack.push("[")
                            t.forEach {
                                stack.push(it)
                            }
                            stack.push("]")
                        }
                    }
                }
                in 'a'..'z' -> {
                    var curValue = "" + cur
                    var ii = i + 1
                    var change = false
                    while (ii < exp.length) {
                        if (exp[ii] in 'a'..'z') {
                            curValue += exp[ii]
                            change = true
                        } else {
                            break
                        }
                        ii++
                    }
                    if (change) {
                        i = ii - 1
                    }
                    stack.push("[")
                    stack.push(curValue)
                    stack.push("]")
                }
                ',' -> {
                    stack.push(cur.toString())
                }
            }
            i++
        }
        val res = mutableListOf<String>()
        while (stack.isNotEmpty()) {
            val p = stack.pop()
            if (p != "[" && p != "]") {
                res.add(0, p)
            }
        }
        return res.sorted()
    }
}
```
### 77
```
Find Minimum in Rotated Sorted Array

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums, return the minimum element of this array.

Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times.

Constraints:
n == nums.length
1 <= n <= 5000
-5000 <= nums[i] <= 5000
All the integers of nums are unique.
nums is sorted and rotated between 1 and n times.
```
##### mine 1 too easy?
```
class Solution {
    fun findMin(nums: IntArray): Int {
        return nums.min()!!
    }
}
```
##### mine 2 faster
```
class Solution {
    fun findMin(nums: IntArray): Int {
        val a = nums[0]
        var i = 1
        while (i < nums.size) {
            if(a > nums[i]){
                return nums[i]
            }
            i++
        }
        return a
    }
}
```
##### binary search problem
```
class Solution {
    fun findMin(nums: IntArray): Int {
        var start = 0
        var end = nums.size - 1
        while (start < end) {
            if(nums[start] < nums[end]) return nums[start]
            val mid = (start + end) / 2
            if(nums[mid] >= nums[start]) {
                start = mid + 1
            } else {
                end = mid
            }
        }
        return nums[start]
    }
}
```
### 78
```
Best Time to Buy and Sell Stock with Cooldown

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

Example:
Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
```
##### mine
```
// buy[i] = max(buy[i-1] , cool[i-1] - p) 卖后不能马上买，所以和sell无关
// sell[i] = max(buy[i-1] + p , sell[i-1]) i - 1 为cool，则i不能卖，故和cool无关
// cool[i] = max(cool[i-1], sell[i-1]) cool[i] 必然 <= sell[i] , 因为sell[i]有可能在i时sell => cool[i] = sell[i-1]
// buy[i] = max(buy[i-1], sell[i-2]-p)
// sell[i] = max(buy[i-1] + p , sell[i-1])
class Solution {
    fun maxProfit(prices: IntArray): Int {
        if (prices.size < 2) return 0
        if (prices.size < 3) return if (prices[0] < prices[1]) prices[1] - prices[0] else 0
        val buy = IntArray(prices.size)
        val sell = IntArray(prices.size)
        buy[0] = -prices[0]
        buy[1] = if (prices[0] < prices[1]) -prices[0] else -prices[1]
        sell[0] = 0
        sell[1] = if (prices[0] < prices[1]) prices[1] - prices[0] else 0
        var i = 2
        while (i < prices.size) {
            buy[i] = Math.max(buy[i - 1], sell[i - 2] - prices[i])
            sell[i] = Math.max(buy[i - 1] + prices[i], sell[i - 1])
            i++
        }
        return sell[prices.size - 1]
    }
}
```
##### like
```
class Solution {
    fun maxProfit(prices: IntArray): Int {
        var sell = 0
        var buy = Int.MIN_VALUE
        var preSell = 0
        var preBuy = buy
        for(price in prices) {
            preBuy = buy
            buy = Math.max(preBuy, preSell - price)
            preSell = sell
            sell = Math.max(preBuy + price, preSell)
        }
        return sell
    }
}
```
### 79
```
Best Time to Buy and Sell Stock

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.

Example 2:
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
```
##### mine
```
class Solution {
    fun maxProfit(prices: IntArray): Int {
        var cost = Int.MAX_VALUE
        var pro = 0
        for(price in prices){
            cost = Math.min(cost, price)
            pro = Math.max(pro, price - cost)
        }
        return pro
    }
}
```
### 80
```
Best Time to Buy and Sell Stock II

Say you have an array prices for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.

Example 2:
Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

Constraints:
1 <= prices.length <= 3 * 10 ^ 4
0 <= prices[i] <= 10 ^ 4
```
##### mine
```
class Solution {
    fun maxProfit(prices: IntArray): Int {
        if(prices.size < 2) return 0
        val buy = IntArray(prices.size)
        val sell = IntArray(prices.size)
        buy[0] = -prices[0]
        sell[0] = 0
        var i = 1
        while (i < prices.size) {
            buy[i] = Math.max(sell[i-1] - prices[i], buy[i-1])
            sell[i] = Math.max(sell[i-1], buy[i-1] + prices[i])
            i++
        }
        return sell[prices.size - 1]
    }
}
```
### 81
```
Binary Tree Maximum Path Sum

Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any node sequence from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:
Input: root = [1,2,3]
Output: 6

Example 2:
Input: root = [-10,9,20,null,null,15,7]
Output: 42

Constraints:
The number of nodes in the tree is in the range [0, 3 * 104].
-1000 <= Node.val <= 1000
```
##### mine slow
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
    fun maxPathSum(root: TreeNode?): Int {
        if (root == null) return 0
        val arr = intArrayOf(Int.MIN_VALUE)
        maxPathSumHelp(root, arr)
        return arr[0]
    }
    
    fun maxPathSumHelp(node: TreeNode, resArr: IntArray): Int {
        var res = node.`val`
        var left = 0
        if (node.left != null) {
            left = maxPathSumHelp(node.left!!, resArr)
        }
        var right = 0
        if (node.right != null) {
            right = maxPathSumHelp(node.right!!, resArr)
        }
        if (!(left < 0 && right < 0)) {
            res += Math.max(left, right)
        }
        val temp = if (node.`val` >= 0) {
            if (left < 0 && right < 0)
                node.`val`
            else if (left > 0 && right > 0) {
                node.`val` + left + right
            } else {
                node.`val` + Math.max(left, right)
            }
        } else {
            var r = node.`val`
            if(left + node.`val` > 0) {
                r += left
            }
            if(right + node.`val` > 0) {
                r += right
            }
            r
        }
        resArr[0] = Math.max(temp, resArr[0])
        return res
    }
}
```
##### like slow
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
    var maxValue = 0

    fun maxPathSum(root: TreeNode?): Int {
        maxValue = Int.MIN_VALUE
        maxPathDown(root)
        return maxValue
    }

    private fun maxPathDown(node: TreeNode?): Int {
        if (node == null) return 0
        val left = Math.max(0, maxPathDown(node.left))
        val right = Math.max(0, maxPathDown(node.right))
        maxValue = Math.max(maxValue, left + right + node.`val`)
        return Math.max(left, right) + node.`val`
    }
}
```
### 82
```
Average of Levels in Binary Tree

Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.

Example 1:
Input:
    3
   / \
  9  20
    /  \
   15   7
Output: [3, 14.5, 11]
Explanation:
The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on level 2 is 11. Hence return [3, 14.5, 11].

Note:
The range of node's value is in the range of 32-bit signed integer.
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
    fun averageOfLevels(root: TreeNode?): DoubleArray {
        if (root == null) return doubleArrayOf()
        val res = mutableListOf<Double>()
        val queue = ArrayDeque<TreeNode>()
        var size = 1
        var preSize = 1
        var cur = 0L
        queue.addLast(root)
        while (queue.isNotEmpty()) {
            val node = queue.removeFirst()
            cur += node.`val`

            if (node.left != null) {
                queue.addLast(node.left!!)
            }
            if (node.right != null) {
                queue.addLast(node.right!!)
            }
            size--
            if (size == 0) {
                res.add(cur / preSize.toDouble())
                cur = 0L
                size = queue.size
                preSize = size
            }
        }
        return res.toDoubleArray()
    }
}
```
### 83
```
Making File Names Unique

Given an array of strings names of size n. You will create n folders in your file system such that, at the ith minute, you will create a folder with the name names[i].

Since two files cannot have the same name, if you enter a folder name which is previously used, the system will have a suffix addition to its name in the form of (k), where, k is the smallest positive integer such that the obtained name remains unique.

Return an array of strings of length n where ans[i] is the actual name the system will assign to the ith folder when you create it.

Example 1:
Input: names = ["pes","fifa","gta","pes(2019)"]
Output: ["pes","fifa","gta","pes(2019)"]
Explanation: Let's see how the file system creates folder names:
"pes" --> not assigned before, remains "pes"
"fifa" --> not assigned before, remains "fifa"
"gta" --> not assigned before, remains "gta"
"pes(2019)" --> not assigned before, remains "pes(2019)"

Example 2:
Input: names = ["gta","gta(1)","gta","avalon"]
Output: ["gta","gta(1)","gta(2)","avalon"]
Explanation: Let's see how the file system creates folder names:
"gta" --> not assigned before, remains "gta"
"gta(1)" --> not assigned before, remains "gta(1)"
"gta" --> the name is reserved, system adds (k), since "gta(1)" is also reserved, systems put k = 2. it becomes "gta(2)"
"avalon" --> not assigned before, remains "avalon"

Example 3:
Input: names = ["onepiece","onepiece(1)","onepiece(2)","onepiece(3)","onepiece"]
Output: ["onepiece","onepiece(1)","onepiece(2)","onepiece(3)","onepiece(4)"]
Explanation: When the last folder is created, the smallest positive valid k is 4, and it becomes "onepiece(4)".

Example 4:
Input: names = ["wano","wano","wano","wano"]
Output: ["wano","wano(1)","wano(2)","wano(3)"]
Explanation: Just increase the value of k each time you create folder "wano".

Example 5:
Input: names = ["kaido","kaido(1)","kaido","kaido(1)"]
Output: ["kaido","kaido(1)","kaido(2)","kaido(1)(1)"]
Explanation: Please note that system adds the suffix (k) to current name even it contained the same suffix before.

Constraints:
1 <= names.length <= 5 * 10^4
1 <= names[i].length <= 20
names[i] consists of lower case English letters, digits and/or round brackets.
```
##### mine
```
class Solution {
    fun getFolderNames(names: Array<String>): Array<String> {
        val indexMap = mutableMapOf<String, Pair<Int, MutableSet<Int>>>()
        for (i in names.indices) {
            names[i] = names[i].findNameAndIndex(indexMap).run {
                if (second == 0) first else "$first($second)"
            }
        }
        return names
    }

    fun String.findNameAndIndex(map: MutableMap<String, Pair<Int, MutableSet<Int>>>): Pair<String, Int> {
        val lastLeftIndex = lastIndexOf('(')
        val lastRightIndex = lastIndexOf(')')
        if (lastLeftIndex == -1 || lastRightIndex == -1 || lastRightIndex < lastLeftIndex) {
            return simpleAdd(this, map)
        } else {
            val name = substring(0, lastLeftIndex)
            val numStr = substring(lastLeftIndex + 1, lastRightIndex)
            try {
                val num = numStr.toInt()
                return if (num == 0) {
                    simpleAdd(this, map)
                } else {
                    if (map.containsKey(name)) {
                        if (map[name]!!.second.contains(num)) {
                            simpleAdd(this, map, true)
                        } else {
                            map[name]!!.second.add(num)
                            Pair(name, num)
                        }
                    } else {
                        map[name] = Pair(-1, mutableSetOf(num))
                        Pair(name, num)
                    }
                }
            } catch (e: NumberFormatException) {
            }
            return simpleAdd(this, map)
        }
    }

    fun simpleAdd(
        name: String,
        map: MutableMap<String, Pair<Int, MutableSet<Int>>>,
        contained: Boolean = false
    ): Pair<String, Int> {
        if (map.containsKey(name)) {
            var nextIndex = map[name]!!.first + 1
            while (map[name]!!.second.contains(nextIndex) || (nextIndex == 0 && contained)) {
                nextIndex++
            }
            map[name]!!.second.add(nextIndex)
            map[name] = Pair(nextIndex, map[name]!!.second)
        } else {
            val value = if (contained) 1 else 0
            map[name] = Pair(value, if (contained) mutableSetOf(0, 1) else mutableSetOf(0))
        }
        return Pair(name, map[name]!!.first)
    }
}
```
### 84
```
 Video Stitching

 You are given a series of video clips from a sporting event that lasted T seconds.  These video clips can be overlapping with each other and have varied lengths.

Each video clip clips[i] is an interval: it starts at time clips[i][0] and ends at time clips[i][1].  We can cut these clips into segments freely: for example, a clip [0, 7] can be cut into segments [0, 1] + [1, 3] + [3, 7].

Return the minimum number of clips needed so that we can cut the clips into segments that cover the entire sporting event ([0, T]).  If the task is impossible, return -1.

Example 1:
Input: clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
Output: 3
Explanation: 
We take the clips [0,2], [8,10], [1,9]; a total of 3 clips.
Then, we can reconstruct the sporting event as follows:
We cut [1,9] into segments [1,2] + [2,8] + [8,9].
Now we have segments [0,2] + [2,8] + [8,10] which cover the sporting event [0, 10].

Example 2:
Input: clips = [[0,1],[1,2]], T = 5
Output: -1
Explanation: 
We can't cover [0,5] with only [0,1] and [1,2].

Example 3:
Input: clips = [[0,1],[6,8],[0,2],[5,6],[0,4],[0,3],[6,7],[1,3],[4,7],[1,4],[2,5],[2,6],[3,4],[4,5],[5,7],[6,9]], T = 9
Output: 3
Explanation: 
We can take clips [0,4], [4,7], and [6,9].

Example 4:
Input: clips = [[0,4],[2,8]], T = 5
Output: 2
Explanation: 
Notice you can have extra video after the event ends.

Constraints:
1 <= clips.length <= 100
0 <= clips[i][0] <= clips[i][1] <= 100
0 <= T <= 100
```
##### mine
```
class Solution {
    fun videoStitching(clips: Array<IntArray>, T: Int): Int {
        val res = videoStitchingHelper(clips, 0, T)
        return if (res == 0) -1 else res
    }

    fun videoStitchingHelper(clips: Array<IntArray>, start: Int, end: Int): Int {
        val begins = clips.filter {
            it[0] <= start
        }
        val ends = clips.filter {
            it[1] >= end
        }
        if (begins.isEmpty() || ends.isEmpty()) return 0
        val startNew = begins.maxBy {
            it[1]
        }
        if (startNew!![1] >= end) return 1
        val endNew = ends.minBy {
            it[0]
        }
        if (endNew!![0] <= start) return 1
        if(startNew[1] >= endNew[0]) return 2
        val child = videoStitchingHelper(clips.toMutableList().apply {
            this.removeAll {
                (it[0] <= start && it[1] <= startNew[1]) || (it[1] >= end && it[0] >= endNew[0])
            }
        }.toTypedArray(), startNew[1], endNew[0])
        return if(child > 0) child + 2 else 0
    }
}
```
##### like
```
public int videoStitching(int[][] clips, int T) {
  int res = 0;
  Arrays.sort(clips, new Comparator<int[]>() {
    public int compare(int[] a, int[] b) { return a[0] - b[0]; }
  });
  for (int i = 0, st = 0, end = 0; st < T; st = end, ++res) {
    for (; i < clips.length && clips[i][0] <= st; ++i)
      end = Math.max(end, clips[i][1]);
    if (st == end) return -1;
  }
  return res;
}
```
### 85
```
Lexicographical Numbers

Given an integer n, return 1 - n in lexicographical order.

For example, given 13, return: [1,10,11,12,13,2,3,4,5,6,7,8,9].

Please optimize your algorithm to use less time and space. The input size may be as large as 5,000,000.
```
##### mine 1
```
class Solution {
    fun lexicalOrder(n: Int): List<Int> {
        val res = mutableListOf<String>()
        for (i in 1..n) {
            res.add("$i")
        }
        return res.sorted().map {
            it.toInt()
        }
    }
}
```
##### like
```
class Solution {
    fun lexicalOrder(n: Int): List<Int> {
        val res = mutableListOf<Int>()
        for (i in 1..9) {
            lexicalOrderHelper(i, n, res)
        }
        return res
    }

    fun lexicalOrderHelper(cur: Int, n: Int, list: MutableList<Int>) {
        if(cur > n) return
        list.add(cur)
        for(i in 0..9){
            val value = 10 * cur + i
            if(value > n) return
            lexicalOrderHelper(value, n, list)
        }
    }
}
```
### 86
```
Three Equal Parts

Given an array A of 0s and 1s, divide the array into 3 non-empty parts such that all of these parts represent the same binary value.

If it is possible, return any [i, j] with i+1 < j, such that:

A[0], A[1], ..., A[i] is the first part;
A[i+1], A[i+2], ..., A[j-1] is the second part, and
A[j], A[j+1], ..., A[A.length - 1] is the third part.
All three parts have equal binary value.
If it is not possible, return [-1, -1].

Note that the entire part is used when considering what binary value it represents.  For example, [1,1,0] represents 6 in decimal, not 3.  Also, leading zeros are allowed, so [0,1,1] and [1,1] represent the same value.

Example 1:
Input: [1,0,1,0,1]
Output: [0,3]

Example 2:
Input: [1,1,0,1,1]
Output: [-1,-1]
 
Note:
3 <= A.length <= 30000
A[i] == 0 or A[i] == 1
```
##### mine 1 Time Limit Exceeded
```
class Solution {
    fun threeEqualParts(A: IntArray): IntArray {
        for(i in A.indices) {
            for(j in A.indices.reversed()) {
                if(i < j && threeEqualPartsHelp(A, i, j)){
                    return intArrayOf(i, j)
                }
            }
        }
        return intArrayOf(-1, -1)
    }

    fun threeEqualPartsHelp(arr: IntArray, startIndex: Int, endIndex: Int): Boolean {
        var bs = 0
        while (bs <= startIndex) {
            if (arr[bs] != 0) break
            bs++
        }
        val be = startIndex

        var ms = startIndex + 1
        while (ms <= endIndex - 1) {
            if (arr[ms] != 0) break
            ms++
        }
        val me = endIndex - 1

        var es = endIndex
        while (es <= arr.size - 1) {
            if (arr[es] != 0) break
            es++
        }
        val ee = arr.size - 1

        if (bs == startIndex && ms == endIndex - 1 && es == arr.size - 1) return true

        val step = ee - es
        if (step == me - ms && step == be - bs) {
            var isTrue = true
            for (i in 0 until step) {
                if (!(arr[i + es] == arr[i + ms] && arr[i + es] == arr[i + bs])) {
                    isTrue = false
                    break
                }
            }
            return isTrue
        }
        return false
    }
}
```
##### like
```
class Solution {
    fun threeEqualParts(A: IntArray): IntArray {
        if (A.size < 3) return intArrayOf(-1, -1)
        var oneCount = 0
        for (i in A) {
            if (i == 1) oneCount++
        }

        if (oneCount == 0) return intArrayOf(0, A.size - 1)

        if (oneCount % 3 != 0) return intArrayOf(-1, -1)

        val k = oneCount / 3

        var start = 0
        for (i in A) {
            if (i == 1) {
                break
            }
            start++
        }

        var count = 0
        var a = 0
        for (i in A) {
            if (i == 1) {
                count++
            }

            if (count == k + 1) {
                break
            }
            a++
        }
        var mid = a

        count = 0
        a = 0
        for (i in A) {
            if (i == 1) {
                count++
            }

            if (count == k * 2 + 1) {
                break
            }
            a++
        }
        var end = a
        while (end < A.size && A[start] == A[mid] && A[mid] == A[end]) {
            start++
            mid++
            end++
        }

        if (end == A.size) return intArrayOf(start - 1, mid)

        return intArrayOf(-1, -1)
    }
}
```
### 87
```
Maximum Number of Vowels in a Substring of Given Length

Given a string s and an integer k.

Return the maximum number of vowel letters in any substring of s with length k.

Vowel letters in English are (a, e, i, o, u).

Example 1:
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.

Example 2:
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.

Example 3:
Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.

Example 4:
Input: s = "rhythms", k = 4
Output: 0
Explanation: We can see that s doesn't have any vowel letters.

Example 5:
Input: s = "tryhard", k = 4
Output: 1

Constraints:
1 <= s.length <= 10^5
s consists of lowercase English letters.
1 <= k <= s.length
```
### mine 1 slow
```
class Solution {
    fun maxVowels(s: String, k: Int): Int {
        var res = 0
        val list = LinkedList<Char>()
        var temp = 0
        for (c in s) {
            if (list.size < k) {
                list.add(c)
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                    temp++
                }
            } else {
                val pre = list.removeAt(0)
                list.add(c)
                if (pre == 'a' || pre == 'e' || pre == 'i' || pre == 'o' || pre == 'u') {
                    temp--
                }
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                    temp++
                }
            }
            res = Math.max(temp, res)
        }
        return res
    }
}
```
### mine 2 slow
```
class Solution {
    fun maxVowels(s: String, k: Int): Int {
        var res = 0
        var temp = 0
        val set = mutableSetOf<Char>().apply { 
            add('a')
            add('e')
            add('i')
            add('o')
            add('u')
        }
        for (i in s.indices) {
            val c = s[i]
            if (set.contains(c)) {
                temp++
            }
            if (i >= k) {
                val pre = s[i - k]
                if (set.contains(pre)) {
                    temp--
                }
            }
            res = Math.max(temp, res)
        }
        return res
    }
}
```
### 88
```
Design Linked List

Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node.
If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement the MyLinkedList class:

MyLinkedList() Initializes the MyLinkedList object.
int get(int index) Get the value of the indexth node in the linked list. If the index is invalid, return -1.
void addAtHead(int val) Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
void addAtTail(int val) Append a node of value val as the last element of the linked list.
void addAtIndex(int index, int val) Add a node of value val before the indexth node in the linked list. If index equals the length of the linked list, the node will be appended to the end of the linked list. If index is greater than the length, the node will not be inserted.
void deleteAtIndex(int index) Delete the indexth node in the linked list, if the index is valid.

Example 1:
Input
["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
[[], [1], [3], [1, 2], [1], [1], [1]]
Output
[null, null, null, null, 2, null, 3]
Explanation
MyLinkedList myLinkedList = new MyLinkedList();
myLinkedList.addAtHead(1);
myLinkedList.addAtTail(3);
myLinkedList.addAtIndex(1, 2);    // linked list becomes 1->2->3
myLinkedList.get(1);              // return 2
myLinkedList.deleteAtIndex(1);    // now the linked list is 1->3
myLinkedList.get(1);              // return 3
 

Constraints:
0 <= index, val <= 1000
Please do not use the built-in LinkedList library.
At most 2000 calls will be made to get, addAtHead, addAtTail,  addAtIndex and deleteAtIndex.
```
##### mine
```
class MyLinkedList() {

    /** Initialize your data structure here. */
    class Node(var `val`: Int, var next: Node?)

    private var root = Node(-1, null)

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    fun get(index: Int): Int {
        var cur = root.next
        var count = 0
        while (cur != null) {
            if (count == index) return cur.`val`
            cur = cur.next
            count++
        }
        return -1
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    fun addAtHead(`val`: Int) {
        val newNode = Node(`val`, root.next)
        root.next = newNode
    }

    /** Append a node of value val to the last element of the linked list. */
    fun addAtTail(`val`: Int) {
        val newNode = Node(`val`, null)
        var cur = root.next
        if (cur == null) {
            root.next = newNode
        } else {
            while (cur!!.next != null) {
                cur = cur.next
            }
            cur.next = newNode
        }
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    fun addAtIndex(index: Int, `val`: Int) {
        if (index == 0) addAtHead(`val`)

        var cur = root.next
        var pre = root
        var count = 0
        while (cur != null) {
            if (count == index) break
            pre = cur
            cur = cur.next
            count++
        }

        if (count == index) {
            val newNode = Node(`val`, cur)
            pre.next = newNode
        }
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    fun deleteAtIndex(index: Int) {
        var cur = root.next
        var pre = root
        var count = 0
        while (cur != null) {
            if (count == index) break
            pre = cur
            cur = cur.next
            count++
        }
        if (cur != null) {
            pre.next = cur.next
            cur.next = null
        }
    }

}
```
### 89
```
Reformat Date

Given a date string in the form Day Month Year, where:
    Day is in the set {"1st", "2nd", "3rd", "4th", ..., "30th", "31st"}.
    Month is in the set {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}.
    Year is in the range [1900, 2100].
Convert the date string to the format YYYY-MM-DD, where:
    YYYY denotes the 4 digit year.
    MM denotes the 2 digit month.
    DD denotes the 2 digit day.

Example 1:
Input: date = "20th Oct 2052"
Output: "2052-10-20"

Example 2:
Input: date = "6th Jun 1933"
Output: "1933-06-06"

Example 3:
Input: date = "26th May 1960"
Output: "1960-05-26"

Constraints:
The given dates are guaranteed to be valid, so no error handling is necessary.
```
##### mine
```
class Solution {
    fun reformatDate(date: String): String {
        val monthMap = mutableMapOf(
            Pair("Jan", "01"),
            Pair("Feb", "02"),
            Pair("Mar", "03"),
            Pair("Apr", "04"),
            Pair("May", "05"),
            Pair("Jun", "06"),
            Pair("Jul", "07"),
            Pair("Aug", "08"),
            Pair("Sep", "09"),
            Pair("Oct", "10"),
            Pair("Nov", "11"),
            Pair("Dec", "12")
        )
        val temp = date.split(" ")
        val day = temp[0].toDay() ?: 0
        return if (day < 10) "${temp[2]}-${monthMap[temp[1]]}-0${temp[0].toDay()}" else "${temp[2]}-${monthMap[temp[1]]}-${temp[0].toDay()}"
    }

    fun String.toDay(): Int? {
        return try {
            when {
                contains("st") -> {
                    split("st")[0].toInt()
                }
                contains("nd") -> {
                    split("nd")[0].toInt()
                }
                contains("rd") -> {
                    split("rd")[0].toInt()
                }
                contains("th") -> {
                    split("th")[0].toInt()
                }
                else -> null
            }
        } catch (e: Exception) {
            null
        }
    }
}
```
### 90
```
Minimum Operations to Make Array Equal

You have an array arr of length n where arr[i] = (2 * i) + 1 for all valid values of i (i.e. 0 <= i < n).

In one operation, you can select two indices x and y where 0 <= x, y < n and subtract 1 from arr[x] and add 1 to arr[y] (i.e. perform arr[x] -=1 and arr[y] += 1). The goal is to make all the elements of the array equal. It is guaranteed that all the elements of the array can be made equal using some operations.

Given an integer n, the length of the array. Return the minimum number of operations needed to make all the elements of arr equal.

Example 1:
Input: n = 3
Output: 2
Explanation: arr = [1, 3, 5]
First operation choose x = 2 and y = 0, this leads arr to be [2, 3, 4]
In the second operation choose x = 2 and y = 0 again, thus arr = [3, 3, 3].

Example 2:
Input: n = 6
Output: 9
 
Constraints:
1 <= n <= 10^4
```
##### mine
```
class Solution {
    fun minOperations(n: Int): Int {
        var res = 0
        if(n % 2 == 0) {
            val right = n / 2
            val left = right - 1
            res += 1
            val midValue = right + left + 1
            for(i in 0 until left) {
                res += midValue - (i * 2 + 1)
            }
        } else {
            val mid = n / 2
            val midValue = mid * 2 + 1
            for(i in 0 until mid) {
                res += midValue - (i * 2 + 1)
            }
        }
        return res
    }
}
```
### 91
```
Validate Binary Tree Nodes

You have n binary tree nodes numbered from 0 to n - 1 where node i has two children leftChild[i] and rightChild[i], return true if and only if all the given nodes form exactly one valid binary tree.

If node i has no left child then leftChild[i] will equal -1, similarly for the right child.

Note that the nodes have no values and that we only use the node numbers in this problem.

Example 1:
Input: n = 4, leftChild = [1,-1,3,-1], rightChild = [2,-1,-1,-1]
Output: true

Example 2:
Input: n = 4, leftChild = [1,-1,3,-1], rightChild = [2,3,-1,-1]
Output: false

Example 3:
Input: n = 2, leftChild = [1,0], rightChild = [-1,-1]
Output: false

Example 4:
Input: n = 6, leftChild = [1,-1,-1,4,-1,-1], rightChild = [2,-1,-1,5,-1,-1]
Output: false
 
Constraints:
1 <= n <= 10^4
leftChild.length == rightChild.length == n
-1 <= leftChild[i], rightChild[i] <= n - 1
```
##### mine
```
fun validateBinaryTreeNodes(n: Int, leftChild: IntArray, rightChild: IntArray): Boolean {
        val cnt = IntArray(n)
        for (i in 0 until n) {
            if (leftChild[i] >= 0) {
                cnt[leftChild[i]] += 1
            }
            if (rightChild[i] >= 0) {
                cnt[rightChild[i]] += 1
            }
        }
        var res = 0
        for (i in 0 until n) {
            if (cnt[i] > 1) {
                return false
            }
            if(cnt[i] == 0) res++
        }
        return res == 1
    }
```
### 92
```
Minimum Flips to Make a OR b Equal to c

Given 3 positives numbers a, b and c. Return the minimum flips required in some bits of a and b to make ( a OR b == c ). (bitwise OR operation).
Flip operation consists of change any single bit 1 to 0 or change the bit 0 to 1 in their binary representation.

Example 1:
Input: a = 2, b = 6, c = 5
Output: 3
Explanation: After flips a = 1 , b = 4 , c = 5 such that (a OR b == c)

Example 2:
Input: a = 4, b = 2, c = 7
Output: 1

Example 3:
Input: a = 1, b = 2, c = 3
Output: 0
 
Constraints:
1 <= a <= 10^9
1 <= b <= 10^9
1 <= c <= 10^9
```
##### mine
```
class Solution {
    fun minFlips(a: Int, b: Int, c: Int): Int {
        val d = a xor c
        val e = b xor c
        var sd = Integer.toBinaryString(d)
        var se = Integer.toBinaryString(e)
        var sc = Integer.toBinaryString(c)

        val max = Math.max(sd.length, Math.max(se.length, sc.length))
        if(sd.length < max){
            var a = ""
            var i = sd.length
            while (i < max) {
                a += "0"
                i++
            }
            sd = a + sd
        }
        if(se.length < max){
            var a = ""
            var i = se.length
            while (i < max) {
                a += "0"
                i++
            }
            se = a + se
        }
        if(sc.length < max){
            var a = ""
            var i = sc.length
            while (i < max) {
                a += "0"
                i++
            }
            sc = a + sc
        }

        var i = sd.length - 1
        var j = se.length - 1
        var k = sc.length - 1
        var count = 0
        while (i >= 0 && j >= 0) {
            if(sd[i] == se[j]) {
                if(sd[i] == '1') {
                    val temp = if(k >= 0) sc[k] else '0'
                    if(temp == '1') {
                        count++
                    } else {
                        count += 2
                    }
                }
            } else {
                val temp = if(k >= 0) sc[k] else '0'
                if(temp == '0') {
                    count++
                }
            }
            i--
            j--
            k--
        }
        return count
    }
}
```
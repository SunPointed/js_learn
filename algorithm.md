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
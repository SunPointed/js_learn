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
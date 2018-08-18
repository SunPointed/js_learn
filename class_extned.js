function A(){}

A.prototype.attrA = 'A'

function B(){}

B.prototype = new A()
B.prototype.attrB = 'B'

function C(){}

C.prototype = new B()
C.prototype.attrC = 'C'

var x = new C()

console.log(x instanceof A)
console.log(x instanceof B)
console.log(x instanceof C)

console.log(x.attrA)
console.log(x.attrB)
console.log(x.attrC)
/*
	有很多错误
*/
var WAIT = 1;
var SUCCESS = 2;
var FAIL = 3;

var DEBUG = false;

function MyPromise(execFun){
	var source = this;

	this.success = function(res) {
		if(!source){
			throw 'success source is undefined';
		}

		if(DEBUG){
			console.log('success source -> ');
			console.log(source);
		}

		if(source instanceof MyPromise){
			source.status = SUCCESS;
			source.res = res;
		}else {
			throw 'this is not a MyPromise';
		}
	}

	this.fail = function(reason) {
		if(!source){
			throw 'success source is undefined';
		}

		if(DEBUG){
			console.log('fail source -> ');
			console.log(source);
		}

		if(source instanceof MyPromise){
			source.status = FAIL;
			source.res = reason;
		} else {
			throw 'this is not a MyPromise';
		}
	}

	this.status = WAIT;
	if(execFun.length === 2){
		if(DEBUG){
			console.log('MyPromise this -> ');
			console.log(this);
		}
		execFun.call(this, this.success, this.fail);
	} else {
		throw 'execFun must have two arguments';
	}
}

MyPromise.prototype.then = function(successFun, failFun) {
	if(arguments.length === 0){
		throw 'at least successFun is needed';
	}

	if(DEBUG){
		console.log('then this -> ');
		console.log(this);
	}

	if(this instanceof MyPromise){
		if(this.status === SUCCESS){
			try{
				successFun.call(this, this.res);
			} catch(err) {
				this.res = err;
				this.status = FAIL;
			}
		} else if(this.status === FAIL){
			if(failFun === undefined){
				throw 'failFun is undefined';
			} else {
				try {
					failFun.call(this, this.res);
				} catch(err) {
					this.res = err;
				}
			}
		}

		var pre = this;
		return new MyPromise(function(s, f){
			if(pre.status === SUCCESS){
				s(pre.res);
			} else {
				f(pre.res);
			}
		});
	} else {
		throw 'this is not a MyPromise';
	}
}

MyPromise.prototype.catch = function(failFun) {
	if(arguments.length === 0){
		throw 'failFun is needed';
	}

	if(DEBUG){
		console.log('catch this -> ');
		console.log(this);
	}

	if(this instanceof MyPromise){
		if(this.status === FAIL){
			try {
				failFun.call(this, this.res);
			} catch(err){
				this.res = err;
			}
		}

		var pre = this;
		return new MyPromise(function(s, f){
			f(pre.res);
		});
	} else {
		throw 'this is not a MyPromise';
	}
}
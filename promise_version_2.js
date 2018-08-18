/*
	result like chrome Promise use =>
 */
var WAIT = 1;
var SUCCESS = 2;
var FAIL = 3;

var DEBUG = false;

function MyPromise(execFun){

	this.success = (res) => {

		if(DEBUG){
			console.log('success this -> ');
			console.log(this);
		}

		if(this instanceof MyPromise){
			this.status = SUCCESS;
			this.res = res;
		}else {
			throw 'this is not a MyPromise';
		}
	}

	this.fail = (reason) => {

		if(DEBUG){
			console.log('fail this -> ');
			console.log(this);
		}

		if(this instanceof MyPromise){
			this.status = FAIL;
			this.res = reason;
		} else {
			throw 'this is not a MyPromise';
		}
	}

	this.status = WAIT;
	if(DEBUG){
		console.log('MyPromise this -> ');
		console.log(this);
	}
	execFun.call(this, this.success, this.fail);
}

MyPromise.prototype.then = function(successFun, failFun) {
	if(DEBUG){
		console.log('then this -> ');
		console.log(this);
	}

	if(this instanceof MyPromise){
		var newPromise = new MyPromise(function(){});
		if(this.status === SUCCESS){
			try{
				if(successFun){
					newPromise.res = successFun.call(this, this.res);
				} else {
					newPromise.res = this.res;
				}
				newPromise.status = SUCCESS;
			} catch(err) {
				newPromise.res = err;
				newPromise.status = FAIL;
			}
		} else if(this.status === FAIL){
			try {
				if(failFun){
					newPromise.res = failFun.call(this, this.res);
					newPromise.status = SUCCESS;
				} else {
					newPromise.res = this.res;
					newPromise.status = FAIL;
				}
			} catch(err) {
				newPromise.res = err;
				newPromise.status = FAIL;
			}
		}

		return newPromise;
	} else {
		throw 'this is not a MyPromise';
	}
}

MyPromise.prototype.catch = function(failFun) {
	return this.then(undefined, failFun);
}

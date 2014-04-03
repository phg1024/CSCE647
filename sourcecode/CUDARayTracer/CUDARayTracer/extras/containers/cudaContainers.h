#pragma once

namespace device {

template <typename T, int MaxSize=128>
class stack {
public:
	__device__ stack():topPos(-1){}
	__device__ void push(T val) {
		data[++topPos] = val;
	}
	__device__ T pop() {
		return data[topPos--];
	}
	__device__ T top() const {
		return data[topPos];
	}
	__device__ bool empty() const {
		return topPos < 0;
	}
	__device__ bool full() const {
		return topPos == MaxSize - 1;
	}
	__device__ void clear() {
		topPos = -1;
	}

private:
	int topPos;
	T data[MaxSize];
};

}
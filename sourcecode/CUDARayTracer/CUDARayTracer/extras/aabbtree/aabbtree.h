#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>
using namespace std;

#include "helper_math.h"
#include "tree.hpp"

namespace aabbtree {
	const float3 zero3;
	const float2 zero2;
struct Triangle {
public:
	typedef float3 vert_t;
	__host__ __device__ Triangle(){}
	Triangle(
		float3 v0, float3 v1, float3 v2, 
		float3 n0 = zero3, float3 n1 = zero3, float3 n2 = zero3,
		float2 t0 = zero2, float2 t1 = zero2, float2 t2 = zero2):
	v0(v0), v1(v1), v2(v2),
	n0(n0), n1(n1), n2(n2),
	t0(t0), t1(t1), t2(t2)
	{}
	__host__ __device__ Triangle(const Triangle& tri):
		v0(tri.v0), v1(tri.v1), v2(tri.v2),
		n0(tri.n0), n1(tri.n1), n2(tri.n2),
		t0(tri.t0), t1(tri.t1), t2(tri.t2)
	{}
	__host__ __device__ Triangle(const Triangle&& tri):
		v0(tri.v0), v1(tri.v1), v2(tri.v2),
		n0(tri.n0), n1(tri.n1), n2(tri.n2),
		t0(tri.t0), t1(tri.t1), t2(tri.t2)
	{}
	
	__host__ __device__ Triangle& operator=(const Triangle& tri) {
		v0 = tri.v0; v1 = tri.v1; v2 = tri.v2;
		n0 = tri.n0; n1 = tri.n1; n2 = tri.n2;
		t0 = tri.t0; t1 = tri.t1; t2 = tri.t2;
		return (*this);
	}
	__host__ __device__ Triangle& operator=(Triangle&& tri) {
		v0 = tri.v0; v1 = tri.v1; v2 = tri.v2;
		n0 = tri.n0; n1 = tri.n1; n2 = tri.n2;
		t0 = tri.t0; t1 = tri.t1; t2 = tri.t2;
		return (*this);
	}

	vert_t center() const {
		const float inv3 = 0.333333333333333333333333333333;
		return (v0 + v1 + v2) * inv3;
	}
	bool intersectionTest(const float3& origin, const float3& dest, float &t) {
		return true;
	}

	float3 v0, v1, v2;
	float3 n0, n1, n2;
	float2 t0, t1, t2;
};

struct AABB
{
public:
	enum Axis {
		X = 0, Y, Z
	};

	AABB() {
		minPt = make_float3(FLT_MAX);
		maxPt = make_float3(-FLT_MAX);
	}

	AABB(const Triangle& tri) {
		const float3 bias = make_float3(1e-4);
		minPt = fminf(fminf(tri.v0, tri.v1), tri.v2)-bias;
		maxPt = fmaxf(fmaxf(tri.v0, tri.v1), tri.v2)+bias;
	}

	AABB(Triangle tris[], int count) {
		minPt = make_float3(FLT_MAX);
		maxPt = make_float3(-FLT_MAX);
		for(int i=0;i<count;i++) {
			AABB bb = AABB(tris[i]);
			minPt = fminf(bb.minPt, minPt);
			maxPt = fmaxf(bb.maxPt, maxPt);
		}
	}

	/// @brief intersection test with a finite length ray
	bool intersectTest(const float3& orig, const float3& dir, const float3& invDir);

	float range(Axis axis) const {
		return (&(maxPt - minPt).x)[axis];
	}

	float3 range() const {
		return maxPt - minPt;
	}

	float3 minPt;
	float3 maxPt;
};

static const int MAX_TRIS_PER_NODE = 4;

struct AABBNode_Serial {
	enum NodeType {
		EMPTY_NODE = 0,
		INTERNAL_NODE = 1,
		LEAF_NODE = 2
	};

	__host__ __device__ AABBNode_Serial():type(EMPTY_NODE), ntris(0), leftChild(-1), rightChild(-1) {}
	__host__ __device__ AABBNode_Serial(const AABBNode_Serial& n):
		type(n.type), aabb(n.aabb), ntris(n.ntris), leftChild(n.leftChild), rightChild(n.rightChild) {
			for(int i=0;i<n.ntris;i++) {
				tri[i] = n.tri[i];
			}
	}

	__host__ __device__ AABBNode_Serial& operator=(const AABBNode_Serial& n) {
		type = n.type;
		aabb = n.aabb;
		ntris = n.ntris;
		leftChild = n.leftChild;
		rightChild = n.rightChild;
		for(int i=0;i<ntris;i++) {
			tri[i] = n.tri[i];
		}
	}

	NodeType type;
	AABB aabb;
	Triangle tri[MAX_TRIS_PER_NODE];
	int ntris;

	int leftChild;
	int rightChild;
};

/// @brief a binary tree
struct AABBNode
{
	typedef AABBNode_Serial::NodeType NodeType;

	AABBNode_Serial serialize(int leftIdx, int rightIdx) const {
		AABBNode_Serial n;
		n.type = type;
		n.aabb = aabb;
		memcpy(&(n.tri[0]), tri, sizeof(Triangle)*MAX_TRIS_PER_NODE);
		n.ntris = ntris;
		n.leftChild = leftIdx;
		n.rightChild = rightIdx;
		return n;
	}

	NodeType type;
	AABB aabb;
	Triangle tri[MAX_TRIS_PER_NODE];
	int ntris;

	AABBNode* leftChild;
	AABBNode* rightChild;
};

class AABBTree : public Tree<AABBNode>
{
public:
	AABBTree();
	AABBTree(const vector<Triangle>& tris);
	~AABBTree();

	void printNodeStats();
	bool intersectTest(const float3& origin, const float3& direction, float &t, Triangle& tri);

	vector<AABBNode_Serial> toArray() const;

private:
	AABBNode* buildAABBTree(const vector<Triangle>& tris, int level=0);
	void releaseTree(AABBNode* node);

	size_t nodeCount[3];
	size_t nodeCountLevel[128];
	size_t maxDepth;

private:
	// non-copyable
	AABBTree(const AABBTree&);
	AABBTree& operator=(const AABBTree&);
};

}
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
		unsigned int idx,
		float3 v0, float3 v1, float3 v2//, 
		//float3 n0 = zero3, float3 n1 = zero3, float3 n2 = zero3,
		//float2 t0 = zero2, float2 t1 = zero2, float2 t2 = zero2
		):
	idx(idx),
	v0(v0), v1(v1), v2(v2)//,
	//n0(n0), n1(n1), n2(n2),
	//t0(t0), t1(t1), t2(t2)
	{}
	__host__ __device__ Triangle(const Triangle& tri):
		idx(tri.idx),
		v0(tri.v0), v1(tri.v1), v2(tri.v2)//,
		//n0(tri.n0), n1(tri.n1), n2(tri.n2),
		//t0(tri.t0), t1(tri.t1), t2(tri.t2)
	{}
	__host__ __device__ Triangle(const Triangle&& tri):
		idx(tri.idx),
		v0(tri.v0), v1(tri.v1), v2(tri.v2)//,
		//n0(tri.n0), n1(tri.n1), n2(tri.n2),
		//t0(tri.t0), t1(tri.t1), t2(tri.t2)
	{}
	
	__host__ __device__ Triangle& operator=(const Triangle& tri) {
		idx = tri.idx;
		v0 = tri.v0; v1 = tri.v1; v2 = tri.v2;
		//n0 = tri.n0; n1 = tri.n1; n2 = tri.n2;
		//t0 = tri.t0; t1 = tri.t1; t2 = tri.t2;
		return (*this);
	}
	__host__ __device__ Triangle& operator=(Triangle&& tri) {
		idx = tri.idx;
		v0 = tri.v0; v1 = tri.v1; v2 = tri.v2;
		//n0 = tri.n0; n1 = tri.n1; n2 = tri.n2;
		//t0 = tri.t0; t1 = tri.t1; t2 = tri.t2;
		return (*this);
	}

	vert_t center() const {
		const float inv3 = 0.333333333333333333333333333333;
		return (v0 + v1 + v2) * inv3;
	}
	bool intersectionTest(const float3& origin, const float3& dest, float &t) {
		// not implemented here
		return true;
	}

	float area() {
		float3 e1 = v1-v0, e2 = v2-v0;
		return 0.5 * length(cross(e1, e2));
	}

	unsigned int idx;
	float3 v0, v1, v2;
	//float3 n0, n1, n2;
	//float2 t0, t1, t2;
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
		const float3 bias = make_float3(1e-6);
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

	// clip a triangle's bounding box
	AABB clip(const Triangle& tri) const {
		AABB bb(tri);
		bb.minPt = fmaxf(minPt, bb.minPt);
		bb.maxPt = fminf(maxPt, bb.maxPt);
		return bb;
	}

	bool isPlanar() const {
		return minPt.x == maxPt.x || minPt.y == maxPt.y || minPt.z == maxPt.z;
	}

	/// @brief intersection test with a finite length ray
	bool intersectTest(const float3& orig, const float3& dir, const float3& invDir);

	float range(Axis axis) const {
		return (&(maxPt - minPt).x)[axis];
	}

	float surfaceArea() const {
		float3 r = range();
		return 2.0 * (r.x * r.y + r.y * r.z + r.z * r.x);
	}

	float3 range() const {
		return maxPt - minPt;
	}
	float range(int axis) const {
		return (&(maxPt - minPt).x)[axis];
	}

	float minPos(int axis) const {
		return (&(minPt.x))[axis];
	}
	float maxPos(int axis) const {
		return (&(maxPt.x))[axis];
	}
	float& minPos(int axis) {
		return (&(minPt.x))[axis];
	}
	float& maxPos(int axis) {
		return (&(maxPt.x))[axis];
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

	friend ostream& operator<<(ostream& os, const AABBNode_Serial& n);

	NodeType type;
	unsigned char ntris;	
	unsigned int tri[MAX_TRIS_PER_NODE];	
	int leftChild;
	int rightChild;
	AABB aabb;
};

inline ostream& operator<<(ostream& os, const AABBNode_Serial& n) {
	os << n.type << '\t'
		<< n.aabb.minPt.x << ", " << n.aabb.minPt.y << ", " << n.aabb.minPt.z << '\t' 
		<< n.aabb.maxPt.x << ", " << n.aabb.maxPt.y << ", " << n.aabb.maxPt.z << '\t' 
		<< '(' << n.leftChild << ", " << n.rightChild << ')' << '\t' << n.ntris;
	return os;
}

/// @brief a binary tree
struct AABBNode
{
	typedef AABBNode_Serial::NodeType NodeType;

	AABBNode_Serial serialize(int leftIdx, int rightIdx) const {
		AABBNode_Serial n;
		n.type = type;
		n.aabb = aabb;
		memcpy(&(n.tri[0]), tri, sizeof(unsigned int)*MAX_TRIS_PER_NODE);
		n.ntris = ntris;
		n.leftChild = leftIdx;
		n.rightChild = rightIdx;
		return n;
	}

	NodeType type;
	AABB aabb;
	unsigned int tri[MAX_TRIS_PER_NODE];
	int ntris;

	AABBNode* leftChild;
	AABBNode* rightChild;
};

struct SplittingPlane {
	SplittingPlane(){}
	SplittingPlane(int axis, float pe):axis(axis), pe(pe){}

	bool operator==(const SplittingPlane& sp) { return (axis == sp.axis) && (pe == sp.pe); }

	int axis;
	float pe;
};

enum PlaneSide { Left = -1, Right = 1 };

struct SplitEvent {
	enum Type {
		EndingOnPlane = 0,// primitive lying entirely to the right of the splitting plane
		LyingOnPlane = 1,			// primitive lying on exactly on the plane
		StartingOnPlane = 2,	// primitive lying entirely to the left of the splitting plane
	};

	SplittingPlane p;
	Type type;
	Triangle t;

	SplitEvent(){}
	SplitEvent(const Triangle& t, int axis, float ee, Type type):
		t(t), type(type)
	{
		p = SplittingPlane(axis, ee);
	}

	bool operator<(const SplitEvent& e) {
		return (p.pe < e.p.pe) || (p.pe == e.p.pe && type < e.type);
	}
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

	void SAH(const SplittingPlane& p, const AABB& V, int NL, int NR, int NP, float& CP, PlaneSide& pside);
	AABBNode* buildAABBTree_SAH(const vector<Triangle>& inTris, const SplittingPlane& pprev, int level=0);
	void findBestPlane(	const vector<Triangle>& T, const AABB& V, SplittingPlane& p, float& Cp, PlaneSide& pside);
	void splitVoxel( const AABB& V, const SplittingPlane& p, AABB& VL, AABB& VR );
	void sortTrianglesToVoxel( const vector<Triangle>& T, const SplittingPlane& p, const PlaneSide& pside, 
		vector<Triangle>& TL, vector<Triangle>& TR );

	size_t nodeCount[3];
	size_t nodeCountLevel[128];
	size_t maxDepth;

private:
	// non-copyable
	AABBTree(const AABBTree&);
	AABBTree& operator=(const AABBTree&);

	void releaseTree(AABBNode* node);
};

}
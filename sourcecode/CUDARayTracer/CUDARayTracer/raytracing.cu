#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <helper_math.h>    // includes cuda.h and cuda_runtime_api.h

#include "definitions.h"
#include "utils.h"
#include "randvec.h"
#include "proceduralTexture.cuh"
#include "extras/containers/cudaContainers.h"

__device__ int shadingMode;
__device__ bool circularSpecular = false;
__device__ bool rectangularSpecular = true;
__device__ bool rampedSpecular = false;
__device__ bool cartoonShading = false;
__device__ bool hasReflection = false;
__device__ bool hasRefraction = false;
__device__ bool isRayTracing = false;
__device__ bool isDirectionalLight = false;
__device__ bool isSpotLight = false;
__device__ bool fakeSoftShadow = false;
__device__ bool rainbowLight = false;
__device__ bool jittered = true;

__device__ int envMapIdx;

// texture using tex object
__device__ cudaTextureObject_t tex[64];

__global__ void setParams(int specType, int tracingType, int envmap) {
	envMapIdx = envmap;
	pn.init();

	circularSpecular = false;
	rectangularSpecular = false;
	rampedSpecular = false;
	cartoonShading = false;
	hasReflection = false;
	hasRefraction = false;

	switch( specType ) {
	case 1:
		circularSpecular = true;
		break;
	case 2:
		rectangularSpecular = true;
		break;
	case 3:
		rampedSpecular = true;
		break;
	case 4:
		cartoonShading = true;
		break;
	default:
		break;
	}

	switch( tracingType ) {
	case 0:
		isRayTracing = true;
		jittered = false;
		break;
	case 1:
		isRayTracing = true;
		jittered = true;
		break;
	case 2:
		isRayTracing = false;
		jittered = true;
		break;
	default:
		break;
	}
}

__global__ void bindTexture2(const cudaTextureObject_t *texs, int texCount) {
	for(int i=0;i<texCount;i++){
		tex[i] = texs[i];
	}
}

__device__ bool lightRayIntersectsBoundingBox( Ray r, float3 rdirinv, BoundingBox bb, float t0 = 0.0, float t1 = FLT_MAX ) {
	float tmin, tmax;

	float l1   = (bb.minPt.x - r.origin.x) * rdirinv.x;
	float l2   = (bb.maxPt.x - r.origin.x) * rdirinv.x;
	tmin = fminf(l1,l2);
	tmax = fmaxf(l1,l2);

	l1   = (bb.minPt.y - r.origin.y) * rdirinv.y;
	l2   = (bb.maxPt.y - r.origin.y) * rdirinv.y;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	l1   = (bb.minPt.z - r.origin.z) * rdirinv.z;
	l2   = (bb.maxPt.z - r.origin.z) * rdirinv.z;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	return ((tmax >= tmin) && (tmax >= t0) && tmin < t1);
}

__device__ bool lightRayIntersectsBoundingBox( Ray r, float3 rdirinv, const aabbtree::AABB& bb, float& tmin, float& tmax, float t0 = 0.0, float t1 = FLT_MAX ) {
#if 1
	float l1   = (bb.minPt.x - r.origin.x) * rdirinv.x;
	float l2   = (bb.maxPt.x - r.origin.x) * rdirinv.x;
	tmin = fminf(l1,l2);
	tmax = fmaxf(l1,l2);

	l1   = (bb.minPt.y - r.origin.y) * rdirinv.y;
	l2   = (bb.maxPt.y - r.origin.y) * rdirinv.y;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	l1   = (bb.minPt.z - r.origin.z) * rdirinv.z;
	l2   = (bb.maxPt.z - r.origin.z) * rdirinv.z;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	return ((tmax >= tmin) && (tmax >= t0) && tmin < t1);
#else
	float3 rdirinv = 1.0 / r.dir;
	
	/*
	rdirinv.x = (r.dir.x==0)?FLT_MAX:rdirinv.x;
	rdirinv.y = (r.dir.y==0)?FLT_MAX:rdirinv.y;
	rdirinv.z = (r.dir.z==0)?FLT_MAX:rdirinv.z;
	*/

	float tmin, tmax, tymin, tymax, tzmin, tzmax;

    if( r.dir.x >= 0 )
    {
		tmin = (bb.minPt.x - r.origin.x) * rdirinv.x;
		tmax = (bb.maxPt.x - r.origin.x) * rdirinv.x;
    }
    else
    {
		tmin = (bb.maxPt.x - r.origin.x) * rdirinv.x;
		tmax = (bb.minPt.x - r.origin.x) * rdirinv.x;
    }

    if( r.dir.y >= 0 )
    {
		tymin = (bb.minPt.y - r.origin.y) * rdirinv.y;
		tymax = (bb.maxPt.y - r.origin.y) * rdirinv.y;
    }
    else
    {
		tymin = (bb.maxPt.y - r.origin.y) * rdirinv.y;
		tymax = (bb.minPt.y - r.origin.y) * rdirinv.y;
    }

    if( tmin > tymax || tymin > tmax )
        return false;

    if( tymin > tmin )
        tmin = tymin;

    if( tymax < tmax )
        tmax = tymax;

    if( r.dir.z >= 0 )
    {
		tzmin = (bb.minPt.z - r.origin.z) * rdirinv.z;
		tzmax = (bb.maxPt.z - r.origin.z) * rdirinv.z;
    }
    else
    {
		tzmax = (bb.minPt.z - r.origin.z) * rdirinv.z;
		tzmin = (bb.maxPt.z - r.origin.z) * rdirinv.z;
    }

    if( tmin > tzmax || tzmin > tmax )
        return false;

    if( tzmin > tmin )
        tmin = tzmin;

    if( tzmax < tmax )
        tmax = tzmax;

    // survived all tests
    return ((tmin < FLT_MAX) && (tmax > 0));
#endif
}

// light ray intersection tests
__device__ float lightRayIntersectsSphere( Ray r, d_Shape* shapes, int sid ) {
    float3 pq = r.origin - shapes[sid].p;
    float a = 1.0;
    float b = dot(pq, r.dir);
    float c = dot(pq, pq) - shapes[sid].radius[0] * shapes[sid].radius[0];

    // solve the quadratic equation
    float delta = b*b - a*c;
    if( delta < 0.0 )
    {
        return -1.0;
    }
    else
    {
        delta = sqrt(delta);
        float x0 = -b+delta; // a = 1, no need to do the division
        float x1 = -b-delta;
        float THRES = 1e-3;

        if( x0 < THRES ) {
            return -1.0;
        }
        else
        {
            if( x1 < THRES ) return x0;
            else return x1;
        }
    }
}

__device__ float lightRayIntersectsPlane( Ray r, d_Shape* shapes, int sid ) {
	//if( !lightRayIntersectsBoundingBox(r, shapes[sid].bb ) ) return -1.0;
    float t = 1e10;
    float THRES = 1e-3;
    float3 pq = shapes[sid].p - r.origin;
    float ldotn = dot(shapes[sid].axis[0], r.dir);
    if( abs(ldotn) < 1e-6 ) return -1.0;
    else {
        t = dot(shapes[sid].axis[0], pq) / ldotn;

        if( t >= THRES ) {
            float3 p = r.origin + t * r.dir;

            // compute u, v coordinates
            float3 pp0 = p - shapes[sid].p;
            float u = dot(pp0, shapes[sid].axis[1]);
            float v = dot(pp0, shapes[sid].axis[2]);
            if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return -1.0;
            else {
                return t;
            }
        }
        else return -1.0;
    }
}

__device__ float lightRayIntersectsQuadraticSurface(Ray r, d_Shape* shapes, int sid) {
	if( !lightRayIntersectsBoundingBox(r, 1.0/r.dir, shapes[sid].bb ) ) return -1.0;

	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - shapes[sid].constant2;

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return -1.0;
    }
    else
    {
		delta = sqrt(delta);        
		float inv = 1.0 / a;

		float x0 = (-b-delta)*inv;
		float x1 = (-b+delta)*inv;
        
		const float THRES = 1e-3;
		float t = 1e10;
		if( x0 > THRES ) t = min(x0, t);
		if( x1 > THRES ) t = min(x1, t);
		if( t > 9e9 ) t = -1.0;
		return t;
    }
}

__device__ float rayIntersectsTriangle(Ray r, float3 v0, float3 v1, float3 v2) {
#if 0
	float3 e1, e2;
	e1 = v1 - v0;
	e2 = v2 - v0;

	float3 n = normalize(cross(e1, e2));

	// check if ray and plane are parallel ?
    float NdotRayDirection = dot(n, r.dir);
    if (NdotRayDirection == 0)
        return -1.0; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = dot(n, v0);
    // compute t (equation 3)
	float t = -(dot(n, r.origin-v0)) / NdotRayDirection;

    // check if the triangle is in behind the ray
    if (t < 0)
        return -1.0; // the triangle is behind 
    // compute the intersection point using equation 1
    float3 p = r.origin + t * r.dir;
 
    //
    // Step 2: inside-outside test
    //
 
    float3 c; // vector perpendicular to triangle's plane
 
    // edge 0
    float3 edge0 = v1 - v0;
    float3 vp0 = p - v0;
    c = cross(edge0, vp0);
    if (dot(n, c) < 0)
        return -1.0; // P is on the right side
 
    // edge 1
    float3 edge1 = v2 - v1;
    float3 vp1 = p - v1;
    c = cross(edge1, vp1);
    if (dot(n, c) < 0)
        return -1.0; // P is on the right side
 
    // edge 2
    float3 edge2 = v0 - v2;
    float3 vp2 = p - v2;
    c = cross(edge2, vp2);
    if (dot(n, c) < 0)
        return -1.0; // P is on the right side;
 
    return t; // this ray hits the triangle
#else
	float3 edge1 = v1 - v0;
	float3 edge2 = v2 - v0;

	float3 pvec = cross(r.dir, edge2);
	float det = dot(edge1, pvec);
	if(det == 0.f)
		return -1.0;
	float inv_det = 1.0f / det;

	float3 tvec = r.origin - v0;
	float u = dot(tvec, pvec) * inv_det;

	float3 qvec = cross(tvec, edge1);
	float v = dot(r.dir, qvec) * inv_det;
	float t = dot(edge2, qvec) * inv_det;

	bool hit = (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f);
	const float EPS = 1e-3;
	if( hit ){ 
		//printf("hit\n"); 
		return t-EPS; 
	}
	else{ 
		//printf("no hit: u = %f, v = %f", u, v);
		return -1.0; 
	}
#endif
}

__device__ __forceinline__ float3 compute_barycentric_coordinates(float3 p, float3 q1, float3 q2, float3 q3) {
	float3 e23 = q3 - q2, e21 = q1 - q2;
	float3 d1 = q1 - p, d2 = q2 - p, d3 = q3 - p;
	float3 oriN = cross(e23, e21);
	float3 n = normalize(oriN);

	float invBTN = 1.0 / dot(oriN, n);
	float3 bcoord;
	bcoord.x = dot(cross(d2, d3), n) * invBTN;
	bcoord.y = dot(cross(d3, d1), n) * invBTN;
	bcoord.z = 1 - bcoord.x - bcoord.y;

	return bcoord;
}

__device__ float lightRayIntersectsTriangles(Ray r, const TriangleMeshInfo& mesh, const aabbtree::AABBNode_Serial& node, int& tri, float tmin = 0.0, float tmax = FLT_MAX) {
	float t = tmax;	
	int hitIdx = -1;
	// leaf node, test all primitives
	for(int i=0;i<node.ntris;i++) {
		int trioffset = node.tri[i] * 3;
		float4 v0, v1, v2;
		v0 = mesh.faces[trioffset];
		v1 = mesh.faces[trioffset+1];
		v2 = mesh.faces[trioffset+2];

		float tmp = rayIntersectsTriangle(r, tofloat3(v0), tofloat3(v1), tofloat3(v2));
		if( tmp > tmin && tmp < t ) {
			t = tmp;
			hitIdx = i;
		}
	}

	if( hitIdx >= 0 ){
		tri = node.tri[hitIdx];
		return t;
	}
	else return -1.0;
}

struct TraverseInfo {
	__device__ TraverseInfo(){}
	__device__ TraverseInfo(int idx, float t0, float t1):idx(idx), tmin(t0), tmax(t1){}
	int idx;
	float tmin, tmax;
};

__device__ float lightRayIntersectsAABBTree_Iterative(Ray r, const TriangleMeshInfo& mesh, int nidx, int& tri) {
	typedef aabbtree::AABBNode_Serial& node_t;

	aabbtree::AABBNode_Serial* tree = mesh.tree;

	float t = FLT_MAX;
	bool hashit = false;

	float3 rdirinv = 1.0 / r.dir;

	device::stack<TraverseInfo, 32> Q;
	Q.push(TraverseInfo(0, 0, FLT_MAX));

	float tmin, tmax;

	while( !Q.empty() ) {
		TraverseInfo info = Q.pop();
		node_t node = tree[info.idx];
		if( node.type == aabbtree::AABBNode_Serial::LEAF_NODE ) {
			// test intersection
			float tmp;
			int tmptri;
			tmp = lightRayIntersectsTriangles(r, mesh, node, tmptri, 0, t);
			if( tmp > 0 && tmp < t ) {
				t = tmp;
				tri = tmptri;
				hashit = true;
			}
		}
		else if( node.type == aabbtree::AABBNode_Serial::INTERNAL_NODE ) {
			// push children to the stack
			int leftIdx = node.leftChild, rightIdx = node.rightChild;

#if 0
			float dl = dot(0.5 * (tree[leftIdx].aabb.minPt + tree[leftIdx].aabb.maxPt) - r.origin, r.dir);
			float dr = dot(0.5 * (tree[rightIdx].aabb.minPt + tree[rightIdx].aabb.maxPt) - r.origin, r.dir);
			
			if( dl < dr ) {
				if( lightRayIntersectsBoundingBox(r, rdirinv, tree[rightIdx].aabb, tmin, tmax, 0, t) ) 
					Q.push(TraverseInfo(rightIdx, tmin, tmax));
				if( lightRayIntersectsBoundingBox(r, rdirinv, tree[leftIdx].aabb, tmin, tmax, 0, t) ) 
					Q.push(TraverseInfo(leftIdx, tmin, tmax));
			}
			else {
				if( lightRayIntersectsBoundingBox(r, rdirinv, tree[leftIdx].aabb, tmin, tmax, 0, t) ) 
					Q.push(TraverseInfo(leftIdx, tmin, tmax));
				if( lightRayIntersectsBoundingBox(r, rdirinv, tree[rightIdx].aabb, tmin, tmax, 0, t) ) 
					Q.push(TraverseInfo(rightIdx, tmin, tmax));
			}
#else
			float tminL, tminR, tmaxL, tmaxR;
			bool hitleft = lightRayIntersectsBoundingBox(r, rdirinv, tree[leftIdx].aabb, tminL, tmaxL, 0, t);
			bool hitright = lightRayIntersectsBoundingBox(r, rdirinv, tree[rightIdx].aabb, tminR, tmaxR, 0, t);
			if( hitleft && hitright ) {
				if( tminL < tmaxL ) { Q.push(TraverseInfo(leftIdx, tminL, tmaxL)); Q.push(TraverseInfo(rightIdx, tminR, tmaxR)); }
				else { Q.push(TraverseInfo(rightIdx, tminR, tmaxR)); Q.push(TraverseInfo(leftIdx, tminL, tmaxL)); }
			}
			else {
				if( hitleft ) Q.push(TraverseInfo(leftIdx, tminL, tmaxL));
				if( hitright ) Q.push(TraverseInfo(rightIdx, tminR, tmaxR));
			}
#endif
		}
	}

	if( hashit ) return t;
	else return -1.0;
}

__device__ float lightRayIntersectsAABBTree(Ray r, const TriangleMeshInfo& mesh, int nidx, int& tri) {

	// traverse the tree to look for a intersection
	aabbtree::AABBNode_Serial* tree = mesh.tree;
	aabbtree::AABBNode_Serial& node = tree[nidx];
	BoundingBox bb;
	bb.minPt = node.aabb.minPt;
	bb.maxPt = node.aabb.maxPt;

	/*
	printf("%d %f %f %f %f %f %f %d %d %d\n", 
		nidx, bb.minPt.x, bb.minPt.y, bb.minPt.z, 
		bb.maxPt.x, bb.maxPt.y, bb.maxPt.z,
		node.type, node.leftChild, node.rightChild);
		*/

	if( node.type == aabbtree::AABBNode_Serial::EMPTY_NODE ) return -1.0;

	float t = FLT_MAX;
	if( !lightRayIntersectsBoundingBox(r, 1.0/r.dir, bb) ) return -1.0;

	if( node.type == aabbtree::AABBNode_Serial::INTERNAL_NODE )
	{			
		bool hashit = false;
		float tleft = -1.0, tright = -1.0;
		int trileft, triright;

		if( node.leftChild != -1 )
		{
			//printf("try left @ %d\n", node.leftChild);
			tleft = lightRayIntersectsAABBTree(r, mesh, node.leftChild, trileft);
		}
		if( node.rightChild != -1 )
		{
			//printf("try right @ %d\n", node.rightChild);
			tright = lightRayIntersectsAABBTree(r, mesh, node.rightChild, triright);
		}

		bool hitleft = tleft>0.0, hitright = tright>0.0;		
		hashit = hitleft || hitright;
		if( hashit ) {
			if( hitleft && hitright ) {
				if( tleft < tright ) {
					tri = trileft;
					return tleft;
				}
				else {
					tri = triright;
					return tright;
				}
			}
			else if( hitleft ) {
				tri = trileft;
				return tleft;
			}
			else {
				tri = triright;
				return tright;
			}
		}
		else return -1.0; 
	}
	else
	{
		return lightRayIntersectsTriangles(r, mesh, node, tri);
	}
}

__device__ float lightRayIntersectsMesh(Ray r, d_Shape* shapes, int sid, int& fid, float3& bcoords) {
	const d_Shape& sp = shapes[sid];

	if( !lightRayIntersectsBoundingBox(r, 1.0/r.dir, sp.bb) ) return -1.0;
	
	bool hit = false;
	float t = FLT_MAX;
	float4 q1, q2, q3;

	// brute force search

	for(int i=0;i<sp.trimesh.nFaces;i++) {
		float4 v1, v2, v3;
		int offset = i*3;
		v1 = sp.trimesh.faces[offset];
		v2 = sp.trimesh.faces[offset+1];
		v3 = sp.trimesh.faces[offset+2];

		float tmp = rayIntersectsTriangle(r, tofloat3(v1), tofloat3(v2), tofloat3(v3));
		if( tmp > 0.0 && tmp < t ) {
			hit = true;
			t = tmp;
			q1 = v1, q2 = v2, q3 = v3;
			fid = i;
		}
	}

	if( hit ){
		bcoords = compute_barycentric_coordinates(r.origin + t * r.dir, tofloat3(q1), tofloat3(q2), tofloat3(q3));
		return t;
	}
	else return -1.0;
}

__device__  __forceinline__ float lightRayIntersectsShape(Ray r, d_Shape* shapes, int sid) {
	switch( shapes[sid].t ) {
	case Shape::PLANE:
		return lightRayIntersectsPlane(r, shapes, sid);
	case Shape::QUADRATICS:
		return lightRayIntersectsQuadraticSurface(r, shapes, sid);
	case Shape::TRIANGLE_MESH:
		int tri;
		//return lightRayIntersectsAABBTree(r, shapes[sid].trimesh, 0, tri, bcoords);
		return lightRayIntersectsAABBTree_Iterative(r, shapes[sid].trimesh, 0, tri);
	default:
		return -1.0;
	}
}

__device__ float lightRayIntersectsShapes(Ray r, d_Shape* shapes, int shapeCount) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
		// hack
        float hitT = lightRayIntersectsShape(r, shapes, i);
		if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

__device__ float lightRayIntersectsShapes2(Ray r, d_Shape* shapes, int shapeCount, int sid) {
	float T_INIT = 1e10;
    // go through a list of shapes and find closest hit
    float t = T_INIT;

	float THRES = 1e-3;

    for(int i=0;i<shapeCount;i++) {
		// hack
		if( i == sid ) continue;
        float hitT = lightRayIntersectsShape(r, shapes, i);
		if( (hitT > -THRES) && (hitT < t) ) {
            t = hitT;
        }
    }

	if( t < T_INIT )
		return t;
	else return -1.0;
}

__device__ float travelPathLengthQuadraticSurface(Ray r, d_Shape* shapes, int sid, float threshold) {
	float3 pq = shapes[sid].p - r.origin;
	float a = dot(r.dir, mul(shapes[sid].m, r.dir));
	float b = -dot(pq, mul(shapes[sid].m, r.dir));
	float c = dot(pq, mul(shapes[sid].m, pq)) - shapes[sid].constant2;

    float delta = b*b - a*c;
    if (delta < 0.0)
    {
		return 0.0;
    }
    else
    {
		delta = sqrt(delta);        
		float inv = 1.0 / a;

		float x0 = (-b-delta)*inv;
		float x1 = (-b+delta)*inv;
        
		const float THRES = 1e-6;
		float t = 1e10;
		if( x0 >= THRES && x1 >= THRES && x0 <= threshold-1e-6 && x1 <= threshold-1e-6 ) {
			return fabs(x0 - x1);
		}
		else if( x0 <= THRES && x1 <= THRES ){
			return 0.0;
		}
		else if( x0 >= threshold-1e-6 && x1 >= threshold-1e-6 )
			return 0.0;
		else if(fmaxf(x0, x1) <= threshold-1e-6) {
			return fmaxf(x0, x1);
		}
		else return 0.0;
    }
}

__device__ float travelPathLength(Ray r, d_Shape* shapes, int shapeCount, int sid, int lid, float dist) {
	float L = 0.0;

    for(int i=0;i<shapeCount;i++) {
		if( i == sid || i == lid ) continue;
		if( shapes[i].t != Shape::PLANE )
			L += travelPathLengthQuadraticSurface(r, shapes, i, dist);
    }
	return L;
}

/*
__device__ bool checkLightVisibility(float3 p, float3 N, d_Shape* shapes, int nShapes, d_Light* lights) {
	switch( lights->t ) {
	case Light::POINT:
	case Light::SPHERE: {
		float dist = length(p - lights->pos);
		Ray r;
		r.origin = p;
		r.dir = normalize(lights->pos - p);
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return t < THRES || t > dist;
	}
	case Light::SPOT: {
	}
	case Light::DIRECTIONAL: {
		Ray r;
		r.origin = p;
		r.dir = -lights->dir;
		float t = lightRayIntersectsShapes(r, shapes, nShapes);

		float THRES = 1e-3;
		return (t < THRES && (dot(N, lights->dir)<0));
	}
	default:
		return false;
	}
}
*/

__device__ bool checkLightVisibility2(float3 l, float3 p, float3 N, d_Shape* shapes, int nShapes, int sid) {
	float dist = length(p - l);

	Ray r;
	r.origin = p;
	if( isDirectionalLight ) r.dir = normalize(l);
	else r.dir = normalize(l - p);
	float t = lightRayIntersectsShapes2(r, shapes, nShapes, sid);

	float THRES = 1e-3;
	return t < THRES || t > dist-THRES;
}

__device__ __forceinline__ float rand(float x, float y){
  float val = sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return val - floorf(val);
}

__device__ float3 texturefunc(int texId, float2 t, float3 p = make_float3(0, 0, 0), int bumpTex=0) {
	switch( texId ) {
	case TextureObject::Chessboard2D:
		return chessboard(t);
	case TextureObject::Chessboard:
		return chessboard3d(p);
	case TextureObject::Julia2D:
		return juliaset((t.x-0.5)*7.0, (t.y-0.5)*3.5);
	case TextureObject::Julia:
		p = p / 10.0;
		return juliaset3d(p.x, p.y, p.z);
	case TextureObject::Perlin2D:
		return perlin2d(t.x, t.y);
	case TextureObject::Perlin: 
		{
			if( bumpTex == 1 ) {
				const float dx = 0.1, dy = 0.1, dz = 0.1;
				float3 p1, p2, p3;
				p1 = make_float3(p.x + dx, p.y, p.z);
				p2 = make_float3(p.x, p.y + dy, p.z);
				p3 = make_float3(p.x, p.y, p.z + dz);
				float h0 = perlin(p).x;
				float h1 = perlin(p1).x;
				float h2 = perlin(p2).x;
				float h3 = perlin(p3).x;
				return normalize(make_float3(h1-h0, h2-h0, h3-h0)/dx+1.0);
			}
			else
				return perlin(p);
		}
	case TextureObject::Marble:
		return marble(p);
	case TextureObject::WoodGrain:
		return woodgrain(p);
	default: {
		if( texId >= TextureObject::Image ) {
			// solid texturing with image
			p = (p + 1.0)*0.5;
			return tofloat3(tex2D<float4>(tex[texId - TextureObject::Image], p.x, p.y));
		}
		else
			// 2d texturing with image
			return tofloat3(tex2D<float4>(tex[texId], t.x, t.y));
			 }
	}

}

__device__ float3 phongShading2(int2 res, float time, int x, int y, 
								float3 v, float3 N, float2 t, Ray r,
								const d_Shape& sp, const d_Material& mater,
								d_Shape* shapes, int nShapes, 
								int* lights, int lightCount, 
								d_Material* mats, int nMats,
								int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
		if( i == sid ) continue;
		const d_Shape& lt = shapes[lights[i]];
		const d_Material& ltmater = mats[lt.materialId];

			// this is a light, create a shadow ray
			float3 lpos;
#if 0
			if( isRayTracing )
				lpos = lt.p;
			else
#endif
				lpos = lt.randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			//calculate Ambient Term:
			float3 Iamb = mater.ambient * ltmater.emission;


			if( isVisible ) {
				float3 L;
				if( isDirectionalLight )
					L = normalize(lpos);
				else
					L = normalize(lpos - v);

				float3 E = normalize(r.origin-v);
				float3 R = normalize(-reflect(L,N));

				float NdotL, RdotE;
				if( mater.normalTex != -1 ) {
					// normal defined in tangent space
					float3 n_normalmap = texturefunc(mater.normalTex, t, v, 1)*2.0-1.0;
					if(mater.normalTex == TextureObject::Perlin) n_normalmap += N;
										
					float3 tangent;
					if( sp.t == Shape::PLANE )
						tangent = sp.axis[1];
					else
						tangent = normalize(sphere_tangent(N));
					float3 bitangent = cross(N, tangent);

					// find the mapping from tangent space to camera space
					mat3 m_t = mat3(tangent, bitangent, N);

					// convert the normal to to camera space
					NdotL = dot(n_normalmap, normalize(m_t*L));		
					RdotE = dot(m_t*R, normalize(m_t*E));
				}
				else {
					NdotL = dot(N, L);
					RdotE = dot(R, E);
				}

				float spotLightFactor = 1.0;
				if( isSpotLight ) {
					float3 ldir = normalize(lpos);
					const float spotlightCutoff = 0.75;
					float cosLL = clamp(dot(ldir, L), spotlightCutoff, 1.0);
					
					// hard spot light
					//spotLightFactor = step(spotlightCutoff, cosLL);

					// soft spot light
					spotLightFactor = (cosLL - spotlightCutoff)/(1.0-spotlightCutoff);
					NdotL *= spotLightFactor;
				}

				//calculate Diffuse Term:
				float3 Idiff;
				float diffuseFactor = max(NdotL, 0.0);

				if( ltmater.diffuseTex != -1 ) {
					// projective light
					float3 lightDir = lt.axis[0], lightU = lt.axis[1], lightV = lt.axis[2];
					float w = lt.radius[0], h = lt.radius[1], d = lt.radius[2];
					float3 lightColor;
					float t = d / dot(L, lightDir);
					if( t > 0.0 ) {
						float3 pl = lt.p + t * L, po = lt.p + d * lightDir;
						float u = dot(pl - po, lightU) / w + 0.5;
						float v = dot(pl - po, lightV) / h + 0.5;
						if( u > 1.0 || u < 0.0 || v > 1.0 || v < 0.0 )
							lightColor = ltmater.emission;
						else
							lightColor = texturefunc(ltmater.diffuseTex, make_float2(u, v));
					}
					else {
						lightColor = ltmater.emission;
					}

					Idiff = clamp(mater.diffuse * lightColor * diffuseFactor, 0.0, 1.0);
				}
				else
					Idiff = clamp(mater.diffuse * ltmater.emission * diffuseFactor, 0.0, 1.0);

				// calculate Specular Term:
				float specFactor = pow(max(RdotE,0.0),0.3 * mater.shininess);
				if( circularSpecular ) specFactor = step(0.8, specFactor);
				if( rampedSpecular ) specFactor = toonify(specFactor, 4);
				if( rectangularSpecular ) {
					float3 pq = r.origin - sp.p;
					float3 uvec = normalize(cross(pq, N));
					float3 vvec = normalize(cross(uvec, N));

					float ufac = dot(R-E, uvec);
					float vfac = dot(R-E, vvec);

					specFactor = filter(ufac, -0.25, 0.25) * filter(vfac, 0.1, 0.5);
				}
				if( isSpotLight ) {
					specFactor *= spotLightFactor;
				}

				float3 Ispec = mater.specular * ltmater.emission
					* specFactor;
				Ispec = clamp(Ispec, 0.0, 1.0);

				if( mater.diffuseTex != -1 ) {					
					float3 Itexture = texturefunc(mater.diffuseTex, t, (v-sp.p)/make_float3(sp.radius[0], sp.radius[1], sp.radius[2]));
					c = c + Itexture * ((Idiff + Ispec) + Iamb);
				}
				else{
					if( cartoonShading )
						c = c + toonify((Idiff + Ispec) + Iamb, 8);
					else 
						c = c + ((Idiff + Ispec) + Iamb);
				}
			}
			else {
				if( mater.diffuseTex != -1 ) {					
					float3 Itexture = texturefunc(mater.diffuseTex, t, (v-sp.p)/make_float3(sp.radius[0], sp.radius[1], sp.radius[2]));
					c = c + Itexture * Iamb;
				}
				else{
					c = c + Iamb;
				}
			}
    }

    return c;
}

__device__ float3 lambertShading2(int2 res, float time, int x, int y, 
								  float3 v, float3 N, float2 t, Ray r, 
								  const d_Shape& sp, const d_Material& mater,
								  d_Shape* shapes, int nShapes, 
								  int* lights, int lightCount, 
								  d_Material* mats, int nMats,
								  int sid) {
    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
		if( i == sid ) continue;
		const d_Shape& lt = shapes[lights[i]];
		const d_Material& ltmater = mats[lt.materialId];
			// this is a light, create a shadow ray
			float3 lpos;
#if 0
			if( isRayTracing )
				lpos = lt.p;
			else
#endif
				lpos = lt.randomPointOnSurface(res, time, x, y);

			// determine if this light is visible
			bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

			//calculate Ambient Term:
			float3 Iamb = mater.ambient * ltmater.emission;

			if( isVisible || fakeSoftShadow ) {
				float3 L;
				if( isDirectionalLight )
					L = normalize(lt.axis[0]);
				else
					L = normalize(lpos - v);
				float3 E = normalize(r.origin - v);

				float NdotL;

				// change the normal with normal map
				if( mater.normalTex != -1 ) {					
					// normal defined in tangent space
					float3 n_normalmap = texturefunc(mater.normalTex, t, v, 1)*2.0-1.0;
					if(mater.normalTex == TextureObject::Perlin) n_normalmap += N;
										
					float3 tangent;
					if( sp.t == Shape::PLANE )
						tangent = sp.axis[1];
					else
						tangent = normalize(sphere_tangent(N));
					float3 bitangent = cross(N, tangent);

					// find the mapping from tangent space to camera space
					mat3 m_t = mat3(tangent, bitangent, N);

					mat3 m_t_inv = m_t.inv();

					NdotL = dot(normalize(m_t_inv*n_normalmap), L);

					// convert the normal to to camera space
					// NdotL = dot(n_normalmap, normalize(m_t*L));
				}
				else {
					NdotL = dot(N, L);
				}

				if( isSpotLight ) {
					float3 ldir = normalize(lpos);
					NdotL *= step(0.75, dot(ldir, L));
				}


				float3 Itexture;
				if( mater.diffuseTex != -1 ) {
					Itexture = texturefunc(mater.diffuseTex, t, (v-sp.p)/make_float3(sp.radius[0], sp.radius[1], sp.radius[2]));
				}
				else Itexture = make_float3(1, 1, 1);

				float diffuseFactor = max(NdotL, 0.0);
				if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

				float3 Idiff;
				if( rainbowLight ) {
					float3 lightColor = mix(make_float3(1, 0.5, 0.5), make_float3(0.5, 1, 0.5), make_float3(0.5, 0.5, 1), diffuseFactor);
					Idiff = clamp(mater.diffuse * lightColor * diffuseFactor, 0.0, 1.0);
				}
				else{
					if( ltmater.diffuseTex != -1 ) {
						// projective light
						float3 lightDir = lt.axis[0], lightU = lt.axis[1], lightV = lt.axis[2];
						float w = lt.radius[0], h = lt.radius[1], d = lt.radius[2];
						float3 lightColor;
						float t = d / dot(L, lightDir);
						if( t > 0.0 ) {
							float3 pl = lt.p + t * L, po = lt.p + d * lightDir;
							float u = dot(pl - po, lightU) / w + 0.5;
							float v = dot(pl - po, lightV) / h + 0.5;
							if( u > 1.0 || u < 0.0 || v > 1.0 || v < 0.0 )
								lightColor = ltmater.emission;
							else
								lightColor = texturefunc(ltmater.diffuseTex, make_float2(u, v));
						}
						else {
							lightColor = ltmater.emission;
						}

						Idiff = clamp(mater.diffuse * lightColor * diffuseFactor, 0.0, 1.0);
					}
					else
						Idiff = clamp(mater.diffuse * ltmater.emission * diffuseFactor, 0.0, 1.0);
				}
								
				float shadowFactor = 1.0;
				if( fakeSoftShadow ) {
					Ray tr;
					tr.origin = v;
					tr.dir = lpos - v;
					float dist = length(tr.dir);
					tr.dir = normalize(tr.dir);
					const float maxLength = 5.0;
					shadowFactor = clamp((maxLength - travelPathLength(tr, shapes, nShapes, sid, i, dist))/maxLength, 0.0f, 1.0f);
				}

				c += Itexture * (Idiff + Iamb) * shadowFactor;
			}
			else {
				float3 Itexture;
				if( mater.diffuseTex != -1 ) {
					Itexture = texturefunc(mater.diffuseTex, t, (v-sp.p)/make_float3(sp.radius[0], sp.radius[1], sp.radius[2]));
				}
				else Itexture = make_float3(1, 1, 1);

				c += Itexture * Iamb;
			}
    }

    return c;
}

__device__ float3 goochShading2(int2 res, float time, int x, int y, 
								float3 v, float3 N, float2 t, Ray r, 
								const d_Shape& sp, const d_Material& mater,
								d_Shape* shapes, int nShapes, 
								int* lights, int lightCount, 
								d_Material* mats, int nMats,
								int sid) {

    float3 c = make_float3(0, 0, 0);

    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
		if( i == sid ) continue;
		d_Shape& lt = shapes[lights[i]];
		
		// cast a shadow ray
		float3 lpos;
		if( isRayTracing )
			lpos = lt.p;
		else
			lpos = lt.randomPointOnSurface(res, time, x, y);

		// determine if this light is visible
		bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

		float3 L;
		if( isDirectionalLight )
			L = normalize(lpos);
		else
			L = normalize(lpos - v);

		float3 E = normalize(r.origin - v);
		float3 R = normalize(-reflect(L,N));
		float NdotL = dot(N, L);

		float3 diffuse;
		if( mater.diffuseTex != -1 ) {
			/*
			diffuse = texture2D (shapes[sid].tex, t).rgb;
			*/
		}
		else diffuse = mater.diffuse;

		float shadowFactor = 1.0;
		if( fakeSoftShadow ) {
			Ray tr;
			tr.origin = v;
			tr.dir = lpos - v;
			float dist = length(tr.dir);
			tr.dir = normalize(tr.dir);
			const float maxLength = 5.0;
			shadowFactor = clamp((maxLength - travelPathLength(tr, shapes, nShapes, sid, i, dist))/maxLength, 0.0f, 1.0f);
		}
		else {
			shadowFactor = isVisible?1.0:0.0;
		}

		float3 Idiff = diffuse * NdotL * shadowFactor;
		float3 kcdiff = fminf(mater.kcool + mater.alpha * Idiff, 1.0f);
		float3 kwdiff = fminf(mater.kwarm + mater.beta * Idiff, 1.0f);
		float3 kfinal = mix(kcdiff, kwdiff, (NdotL+1.0)*0.5);
		// calculate Specular Term:
		float3 Ispec = mater.specular * mater.emission
			* pow(max(dot(R,E),0.0),0.3*mater.shininess) * (isVisible?1.0:0.0);
		Ispec = step(make_float3(0.5, 0.5, 0.5), Ispec);
		// edge effect
		//float EdotN = dot(E, N);
		//if( fabs(EdotN) >= 0.2 ) 
		c = c + clamp(kfinal + Ispec, 0.0, 1.0);
    }
    return clamp(c, 0.0f, 1.0f);
}

__device__ __forceinline__ Hit background() {
	Hit h;
	h.t = -1.0;
	h.objIdx = -1;
	return h;
}

__device__ float3 computeShading2(int2 res, float time, int x, int y, 
								  float3 p, float3 n, float2 t, Ray r, 
								  d_Shape* shapes, int nShapes, 
								  int* lights, int nLights, 
								  d_Material* mats, int nMats,
								  int sid) {
    switch( shadingMode ) {
	case 1:
        return lambertShading2(res, time, x, y, p, n, t, r, shapes[sid], mats[shapes[sid].materialId], shapes, nShapes, lights, nLights, mats, nMats, sid);
	case 2:
        return phongShading2(res, time, x, y, p, n, t, r, shapes[sid], mats[shapes[sid].materialId], shapes, nShapes, lights, nLights, mats, nMats, sid);
	case 3:
        return goochShading2(res, time, x, y, p, n, t, r, shapes[sid], mats[shapes[sid].materialId], shapes, nShapes, lights, nLights, mats, nMats, sid);
	default:
		return make_float3(0, 0, 0);
	}
}

__device__ __forceinline__ Ray generateRay(Camera* cam, float u, float v) {
	Ray r;
	r.origin = cam->pos.data;

    // find the intersection point on the canvas
    float3 canvasCenter = cam->f * cam->dir.data + cam->pos.data;
    float3 pcanvas = canvasCenter + u * cam->w * cam->right.data + v * cam->h * cam->up.data;

    r.dir = normalize(pcanvas - cam->pos.data);

	return r;
}

__device__ Hit rayIntersectsPlane(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsPlane(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
        h.p = r.origin + h.t * r.dir;
        // compute u, v coordinates
        float3 pp0 = h.p - shapes[sid].p;
        float u = dot(pp0, shapes[sid].axis[1]);
        float v = dot(pp0, shapes[sid].axis[2]);
        if( abs(u) > shapes[sid].radius[0] || abs(v) > shapes[sid].radius[1] ) return background();
        else {
            float2 t = clamp(make_float2(u / shapes[sid].radius[0], v / shapes[sid].radius[1]), -1.0, 1.0);
			float scale = 1.0;//0.5 * (shapes[sid].radius[0]/10.0 + shapes[sid].radius[1]/10.0);
			h.tex = t * scale;
			h.n = shapes[sid].axis[0];
			h.objIdx = sid;

			return h;
        }
    }
	else return background();
}

__device__ Hit rayIntersectsQuadraticSurface(Ray r, d_Shape* shapes, int nShapes, int sid) {
	float ti = lightRayIntersectsQuadraticSurface(r, shapes, sid);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		// hit point
		h.p = h.t * r.dir + r.origin;

        // normal at hit point
		h.n = normalize(2.0 * mul(shapes[sid].m, (h.p - shapes[sid].p)) * shapes[sid].constant);

		// apply normal map
		h.tex = spheremap(h.n);
		
		h.objIdx = sid;
        return h;
    }

	else return background();
}

__device__ Hit rayIntersectsTriangleMesh(Ray r, d_Shape* shapes, int nShapes, int sid) {
#if 0
	int faceIdx = -1;
	float3 bcoords;
	float ti = lightRayIntersectsMesh(r, shapes, sid, faceIdx, bcoords);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		
		// hit point
		h.p = h.t * r.dir + r.origin;

        // normal at hit point
		if( shapes[sid].trimesh.normalTex == -1 ) {
			float4 v1, v2, v3;
			getTriangleVertices(shapes[sid].trimesh.faceTex, faceIdx, v1, v2, v3);
			h.n = normalize(cross(tofloat3(v2-v1), tofloat3(v3-v1)));
		}
		else {
			float4 n1, n2, n3;
			getTriangleNormals(shapes[sid].trimesh.normalTex, faceIdx, n1, n2, n3);
			h.n = tofloat3(bcoords.x * n1 + bcoords.y * n2 + bcoords.z * n3);

			// apply normal map

		}
		
		if( shapes[sid].trimesh.texCoordTex == -1 ) {
			h.tex = spheremap(normalize(h.p - shapes[sid].p));
		}
		else {
			float2 t1, t2, t3;	
			getTriangleTextureCoords(shapes[sid].trimesh.texCoordTex, faceIdx, t1, t2, t3);
			h.tex = bcoords.x * t1 + bcoords.y * t2 + bcoords.z * t3;
		}

		h.objIdx = sid;
        return h;
	}
	else return background();
#else
	int tri;
	float3 bcoords;
	const TriangleMeshInfo& mesh = shapes[sid].trimesh;
	float ti = lightRayIntersectsAABBTree_Iterative(r, mesh, 0, tri);
	//float ti = lightRayIntersectsAABBTree(r, shapes[sid].trimesh, 0, tri);
	if( ti > 0.0 ) {
		Hit h;
		h.t = ti;
		
		// hit point
		h.p = h.t * r.dir + r.origin;

		float4 vm0, vm1, vm2;
		int trioffset = tri * 3;
		vm0 = mesh.faces[trioffset];
		vm1 = mesh.faces[trioffset+1];
		vm2 = mesh.faces[trioffset+2];

		float3 v0 = tofloat3(vm0);
		float3 v1 = tofloat3(vm1);
		float3 v2 = tofloat3(vm2);
		float3 bcoords = compute_barycentric_coordinates(h.p, v0, v1, v2);				


        // normal at hit point
		if( mesh.normals == NULL ) {
			h.n = normalize(cross(v1-v0, v2-v0));
		}
		else {			
			float3 n0, n1, n2;
			n0 = mesh.normals[trioffset];
			n1 = mesh.normals[trioffset+1];
			n2 = mesh.normals[trioffset+2];

			h.n = bcoords.x * n0 + bcoords.y * n1 + bcoords.z * n2;
			//printf("%f %f %f %f %f %f\n", bcoords.x, bcoords.y, bcoords.z, h.n.x, h.n.y, h.n.z);
			// apply normal map
		}
		
		if( mesh.texcoords == NULL ) {
			h.tex = spheremap(normalize(h.p - shapes[sid].p));
		}
		else {
			float2 t0, t1, t2;
			t0 = mesh.texcoords[trioffset];
			t1 = mesh.texcoords[trioffset+1];
			t2 = mesh.texcoords[trioffset+2];
			h.tex = bcoords.x * t0 + bcoords.y * t1 + bcoords.z * t2;
		}

		h.objIdx = sid;
        return h;
	}
	else return background();
#endif
}

__device__ __forceinline__ Hit rayIntersectsShape(Ray r, d_Shape* shapes, int nShapes, int sid) {
	switch( shapes[sid].t ) {
	case Shape::PLANE:
		return rayIntersectsPlane(r, shapes, nShapes, sid);
	case Shape::QUADRATICS:
		return rayIntersectsQuadraticSurface(r, shapes, nShapes, sid);
	case Shape::TRIANGLE_MESH:
		return rayIntersectsTriangleMesh(r, shapes, nShapes, sid);
	default:
		return background();
	}
}

__device__ Hit rayIntersectsShapes(Ray r, int nShapes, d_Shape* shapes) {
	Hit h;
	h.t = 1e10;
	h.objIdx = -1;

    for(int i=0;i<nShapes;i++) {
        Hit hit = rayIntersectsShape(r, shapes, nShapes, i);
        if( (hit.t > 0.0) && (hit.t < h.t) ) {
            h = hit;
        }
    }

	return h;
}

__device__ float3 traceRay_simple(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, int* lights, int nMats, d_Material* mats) {
	Hit h = rayIntersectsShapes(r, nShapes, shapes);
	if( h.objIdx == -1 ) {
		// no hit, sample environment map
		if( envMapIdx >= 0 ) {
			float2 t = spheremap(r.dir);
			return tofloat3(tex2D<float4>(tex[envMapIdx], -t.x, t.y));
		}
		else
			return make_float3(0, 0, 0);
	}
	else {
		const d_Material& mater = mats[shapes[h.objIdx].materialId];
		// hit a light
		if( mater.emission.x != 0 || mater.emission.y != 0 || mater.emission.z != 0 )
		{
			return mater.emission;
		}
		else {
			return computeShading2(res, time, x, y, h.p, h.n, h.tex, r, shapes, nShapes, lights, nLights, mats, nMats, h.objIdx);
		}
	}
}

__device__ float3 computeShadow(int2 res, float time, int x, int y, float3 v, float3 N, float2 t, Ray r, d_Shape* shapes, int nShapes, int* lights, int lightCount, d_Material* mats, int nMats, int sid) {
    float3 c = make_float3(0, 0, 0);

	const d_Shape& sp = shapes[sid];
	const d_Material& mater = mats[sp.materialId];
    // iterate through all lights
    for(int i=0;i<lightCount;i++) {
		if( i == sid ) continue;
		const d_Shape& lt = shapes[lights[i]];
		// this is a light, create a shadow ray
		float3 lpos = lt.randomPointOnSurface(res, time, x, y);

		// determine if this light is visible
		bool isVisible = checkLightVisibility2(lpos, v, N, shapes, nShapes, i);

		if( isVisible ) {
			float3 L = normalize(lpos - v);
			float3 E = normalize(r.origin - v);

			float NdotL;

			// change the normal with normal map
			if( mater.normalTex != -1 ) {
				/*
				// normal defined in tangent space
				vec3 n_normalmap = normalize(texture2D(shapes[sid].nTex, t).rgb * 2.0 - 1.0);

				vec3 tangent = normalize(sphere_tangent(N));
				vec3 bitangent = cross(N, tangent);

				// find the mapping from tangent space to camera space
				mat3 m_t = transpose(mat3(tangent, bitangent, N));

				// convert the normal to to camera space
				NdotL = dot(n_normalmap, normalize(m_t*L));
				*/
			}
			else {
				NdotL = dot(N, L);
			}

			float3 Itexture;
			if( mater.diffuseTex != -1 ) {
				Itexture = tofloat3(tex2D<float4>(tex[mater.diffuseTex], t.x, t.y));
			}
			else Itexture = make_float3(1, 1, 1);

			float diffuseFactor = max(NdotL, 0.0);
			if( cartoonShading ) diffuseFactor = toonify(diffuseFactor, 8);

			float3 Idiff = clamp(mater.diffuse * mats[shapes[i].materialId].emission * diffuseFactor, 0.0, 1.0);

			c += Itexture * Idiff;
		}
    }

    return c;
}

__device__ float3 traceRay_general(float time, int2 res, int x, int y, Ray r, int nShapes, d_Shape* shapes, int nLights, int* lights, int nMats, d_Material* mats) {
	const int maxBounces = 32;
	float3 accumulatedColor = make_float3(0.0);
	float3 colormask = make_float3(1.0);		// not absorbed color
	const float Wcutoff = 1e-6;

	int pixelIdx = y*res.x+x;

	AbsorptionAndScatteringProp asprop;

	Ray ray = r;
	int bounces = 0;
	for(;bounces<maxBounces;bounces++) {
		if( dot(colormask, colormask) < Wcutoff ) {
			// terminate low weight ray
			accumulatedColor += colormask;
			break;
		}

		thrust::default_random_engine rng( myhash(time) * myhash(pixelIdx) * myhash(bounces) );
		thrust::uniform_real_distribution<float> uniformDistribution(0,1);

		Hit h = rayIntersectsShapes(ray, nShapes, shapes);		
		float3 color = make_float3(0.0f);

		if( h.objIdx == -1 ) {	
			// no hit, sample environment map
			if( envMapIdx >= 0 ) {
				float2 t = spheremap(ray.dir);
				colormask *= tofloat3(tex2D<float4>(tex[envMapIdx], -t.x, t.y));
			}
			else {
				const float3 bgcolor = make_float3(.90, .95, .975);
				colormask *= bgcolor;
			}

			accumulatedColor += colormask;
			break;
		}

		const d_Material& mater = mats[shapes[h.objIdx].materialId];

		// process scattering and absorption
		const float ZERO_ABSOPTION_THRESHOLD = 0.00001;
		if ( asprop.reducedScatteringCoeffs > 0 || dot(asprop.absortionCoeffs, asprop.absortionCoeffs) >= ZERO_ABSOPTION_THRESHOLD ) {
			float randomFloatForScatteringDistance = uniformDistribution(rng);
			float scatteringDistance = -log(randomFloatForScatteringDistance) / asprop.reducedScatteringCoeffs;
			if (scatteringDistance < h.t) {
				// printf("SS: %f %f %f %f\n", asprop.reducedScatteringCoeffs, asprop.absortionCoeffs.x, asprop.absortionCoeffs.y, asprop.absortionCoeffs.z);
				// Both absorption and scattering - subsurface scattering
				// Scatter the ray:
				Ray nextRay;
				nextRay.origin = hitpoint(ray, scatteringDistance);
				float2 uv = make_float2(uniformDistribution(rng), uniformDistribution(rng));
				nextRay.dir = calculateRandomDirectionInSphere(uv.x, uv.y); // Isoptropic scattering!

				ray = nextRay;

				// Compute how much light was absorbed along the ray before it was scattered:
				colormask *= transmission(asprop.absortionCoeffs, scatteringDistance);

				// That's it for this iteration!
				continue;
			} else {
				// Just absorption.
				float3 trans = transmission(asprop.absortionCoeffs, h.t);
				colormask *= trans;
			}
		}

		switch( mater.t ) {
		case Material::Emissive:
			{				
				//float maxf = fmaxf(shapes[h.objIdx].material.emission.x, shapes[h.objIdx].material.emission.y);
				//maxf = fmaxf(shapes[h.objIdx].material.emission.z, maxf);				

				// hit a light
				accumulatedColor += mater.emission * colormask;
				colormask *= mater.emission;

				// send it out again
				ray.origin = h.p + 1e-3 * h.n;
				float2 uv = make_float2(uniformDistribution(rng), uniformDistribution(rng));
				
				ray.dir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);
				break;
			}
		case Material::Diffuse:
			{				
				// merged glossy and diffuse
				// change the coefficient Kr to adjust the glossy amount

				// direct lighting
				//float3 shading = computeShadow(res, time, x, y, h.p, h.n, h.tex, ray, shapes, nShapes, lights, nLights, h.objIdx);
				//accumulatedColor += shading * colormask;

				float3 color;
				if( mater.diffuseTex != -1 ) {
					color = texturefunc(mater.diffuseTex, h.tex, h.p);
				}
				else color = mater.diffuse;

				colormask *= color;

				ray.origin = h.p + 1e-3 * h.n;
				ray.dir = reflect(ray.dir, h.n);

				float2 uv = make_float2(uniformDistribution(rng), uniformDistribution(rng));
				
				ray.dir = mix(ray.dir, calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y), mater.Kr);

				ray.dir = normalize(ray.dir);
				break;
			}
		case Material::Specular:
			{
				if( mater.normalTex != -1 ) {
					// modify normal direction
					float3 n_normalmap = texturefunc(mater.normalTex, h.tex, h.p, 1)*2.0-1.0;
										
					float3 tangent;
					if( shapes[h.objIdx].t == Shape::PLANE )
						tangent = shapes[h.objIdx].axis[1];
					else
						tangent = normalize(sphere_tangent(h.n));
					float3 bitangent = cross(h.n, tangent);

					// find the mapping from tangent space to camera space
					mat3 m_t = mat3(tangent, bitangent, h.n);

					mat3 m_t_inv = m_t.inv();

					// change it
					h.n = (m_t_inv*n_normalmap);
				}

				// get the reflected ray
				Ray rf;
				rf.origin = h.p; rf.dir = normalize(reflect(ray.dir, h.n));
				
				double dDn = dot(h.n, ray.dir);
				float3 n1 = dDn<0?h.n:-h.n;
				bool into = dDn<0;
				double nc=1.0, nt=mater.eta;
				double nnt=into?nc/nt:nt/nc, ddn = dot(ray.dir, n1);
				double cos2t=1.0-nnt*nnt*(1-ddn*ddn);

				if( cos2t<0.0 ) {
					// total internal reflection
					rf.origin -= 1e-3 * h.n;
					ray = rf;
				}
				else {
					// choose either reflection or refraction
					Ray rr;
					rr.origin = h.p - 1e-3 * n1; 
					rr.dir = normalize(nnt*ray.dir-(into?1.0:-1.0)*(ddn*nnt+sqrtf(cos2t))*h.n);

					float a = nt-nc, b = nt+nc, R0 = a*a/(b*b), c = 1.0 - (into?-ddn:dot(rr.dir, h.n));
					float Re = R0 + (1-R0)*powf(c, 5.0), Tr =1.0 - Re, P=0.25+0.5*Re, RP = Re/P, TP = Tr/(1.0-P);

					float Xi = uniformDistribution(rng);

					if( Xi < Re ) {
						ray = rf;
						colormask *= mater.specular;
					}
					else {
						float2 uv = make_float2(uniformDistribution(rng), uniformDistribution(rng));
						ray = rr;
						ray.dir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);

						float3 Idiff;
						if( mater.diffuseTex != -1 ) {
							Idiff = texturefunc(mater.diffuseTex, h.tex, h.p);
						}
						else Idiff = mater.diffuse;

						colormask *= Idiff;
					}
				}
				break;
			}
		case Material::Refractive:
			{
				// get the reflected ray
				Ray rf;
				rf.origin = h.p; rf.dir = normalize(reflect(ray.dir, h.n));
				
				double dDn = dot(h.n, ray.dir);
				float3 n1 = dDn<0?h.n:-h.n;
				bool into = dDn<0;
				double nc=1.0, nt=mater.eta;
				double nnt=into?nc/nt:nt/nc, ddn = dot(ray.dir, n1);
				double cos2t=1.0-nnt*nnt*(1-ddn*ddn);

				if( cos2t<0.0 ) {
					// total internal reflection
					rf.origin -= 1e-3 * h.n;
					ray = rf;
				}
				else {
					// choose either reflection or refraction
					Ray rr;
					rr.origin = h.p - 1e-3 * n1; 
					rr.dir = normalize(nnt*ray.dir-(into?1.0:-1.0)*(ddn*nnt+sqrtf(cos2t))*h.n);

					float a = nt-nc, b = nt+nc, R0 = a*a/(b*b), c = 1.0 - (into?-ddn:dot(rr.dir, h.n));
					float Re = R0 + (1-R0)*powf(c, 5.0), Tr =1.0 - Re, P=0.25+0.5*Re, RP = Re/P, TP = Tr/(1.0-P);

					float Xi = uniformDistribution(rng);

					if( Xi < Re ) {
						ray = rf;
						colormask *= mater.specular;
					}
					else {
						ray = rr;
						
						// purturb the refraction direction a little bit
						float2 uv = make_float2(uniformDistribution(rng), uniformDistribution(rng));
						float3 pdir = calculateRandomDirectionInHemisphere(h.n, uv.x, uv.y);

						ray.dir = normalize(mix(ray.dir, pdir, mater.Kf));

						if(into) {
							//printf("in\n");

							float3 Idiff;
							if( mater.diffuseTex != -1 ) {
								Idiff = texturefunc(mater.diffuseTex, h.tex, h.p);
							}
							else Idiff = mater.diffuse;

							asprop.absortionCoeffs = 1.0 - Idiff;
							asprop.reducedScatteringCoeffs = mater.Ks;
						}
						else {
							//printf("out\n");
							asprop.absortionCoeffs = make_float3(0.0);
							asprop.reducedScatteringCoeffs = 0.0;
						}
					}
				}
				break;
			}
		}
	}

	return accumulatedColor;
}

__global__ void clearCumulatedColor(float3* color, int w, int h) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x > w-1 || y > h-1 ) return;

	int idx = y*w+x;
	color[idx] = make_float3(0.0);
}

__global__ void copy2pbo(float3 *color, float3 *pos, int iters, int w, int h, float gamma) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x > w-1 || y > h-1 ) return;

	int idx = y*w+x;

	float3 inC = clamp(color[idx]/iters, 0.0, 1.0);

	inC = pow(inC, 1.0/gamma);

	Color c;
	c.c.data = make_float4(inC, 1.0);

	pos[idx] = make_float3(x, y, c.toFloat());
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float time, float3 *color, Camera* cam, 
						 int nLights, int* lights, 
						 int nShapes, Shape* shapes,
						 int nMaterials, Material* materials,
						 unsigned int width, unsigned int height,
						 int sMode, int AASamples)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	// load scene information into block
	shadingMode = sMode;
	
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ int inMaterialCount;
	__shared__ d_Shape inShapes[16];
	__shared__ int inLights[4];
	__shared__ d_Material inMaterial[64];
	
	inLightsCount = nLights;
	inShapesCount = nShapes;
	inMaterialCount = nMaterials;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid] = lights[tid];	
	if( tid < nMaterials )	inMaterial[tid].init(materials[tid]);
	if( tid < nShapes ){
		inShapes[tid].init(shapes[tid]);
		//inShapes[tid].material = inMaterial[inShapes[tid].materialId];
	}
	
	__syncthreads();

	int2 resolution = make_int2(width, height);

	unsigned int x = blockIdx.x*blockDim.x + tidx;
	unsigned int y = blockIdx.y*blockDim.y + tidy;

	if( x > width - 1 || y > height - 1 ) return;
	
	float3 c = make_float3(0.0);

	float2 offset = generateRandomNumberFromThread2(resolution, time, x, y);

	float u = x + offset.x;
	float v = y + offset.y;
	u = u / (float) width - 0.5;
	v = v / (float) height - 0.5;

	Ray r = generateRay(cam, u, v);

	if( isRayTracing ) {
		c += traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);
	}
	else c += traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);

	// write output vertex
	color[y*width+x] += c;
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program, with load balancing
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace2(float time, float3 *color, Camera* cam, 
						  int nLights, int* lights, 
						  int nShapes, Shape* shapes, 
						  int nMaterials, Material* materials,
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int gx, int gy, int gmx, int gmy)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	// load scene information into block
	shadingMode = sMode;
	
	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ int inMaterialCount;
	__shared__ d_Shape inShapes[16];
	__shared__ int inLights[4];
	__shared__ d_Material inMaterial[64];

	inLightsCount = nLights;
	inShapesCount = nShapes;
	int tid = tidy * blockDim.x + tidx;
	if( tid < nLights )	inLights[tid] = lights[tid];
	if( tid < nMaterials )	inMaterial[tid].init(materials[tid]);
	if( tid < nShapes ){
		inShapes[tid].init(shapes[tid]);
	}
	__syncthreads();
	
		
	int2 resolution = make_int2(width, height);
	for(int gi=0;gi<gmy;gi++) {
		for(int gj=0;gj<gmx;gj++) {

			unsigned int xoffset = gj * gx * blockDim.x;
			unsigned int yoffset = gi * gy * blockDim.y;

			unsigned int x = blockIdx.x*blockDim.x + tidx + xoffset;
			unsigned int y = blockIdx.y*blockDim.y + tidy + yoffset;

			if( x > width - 1 || y > height - 1 ) return;

			float3 c = make_float3(0, 0, 0);

			float2 offset = generateRandomNumberFromThread2(resolution, time, x, y);

			float u = x + offset.x;
			float v = y + offset.y;

			u = u / (float) width - 0.5;
			v = v / (float) height - 0.5;

			Ray r = generateRay(cam, u, v);

			if( isRayTracing ) {
				c = traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);
			}
			else c = traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
			
			c.x = blockIdx.x / (float)gx;
			c.y = blockIdx.y / (float)gy;
			c.z = 0.0;
			
			/*
			c.c.r = threadIdx.x / (float)blockDim.x;
			c.c.g = threadIdx.y / (float)blockDim.y;
			c.c.b = 0.0;
			*/			
#endif

			// write output vertex
			color[y*width+x] += c;
			__syncthreads();
		}
	}
}


__device__ int curb;
__global__ void initCurrentBlock(int v) {
	curb = v;
}

///////////////////////////////////////////////////////////////////////////////
//! main entry of the ray tracing program, with load balancing
///////////////////////////////////////////////////////////////////////////////
__global__ void raytrace3(float time, float3 *color, Camera* cam, 
						  int nLights, int* lights, 
						  int nShapes, Shape* shapes,
						  int nMaterials, Material* materials,
						  unsigned int width, unsigned int height,
						  int sMode, int AASamples, int bmx, int bmy, int ttlb)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tid = tidy * blockDim.x + tidx;

	// load scene information into block
	shadingMode = sMode;	

	__shared__ int inLightsCount;
	__shared__ int inShapesCount;
	__shared__ int inMaterialCount;
	__shared__ d_Shape inShapes[64];
	__shared__ int inLights[4];
	__shared__ d_Material inMaterial[64];

	inLightsCount = nLights;
	inShapesCount = nShapes;
	if( tid < nLights )	inLights[tid] = lights[tid];
	if( tid < nMaterials )	inMaterial[tid].init(materials[tid]);
	if( tid < nShapes ){
		inShapes[tid].init(shapes[tid]);
	}
	__syncthreads();

	__shared__ int currentBlock;
	if( tid == 0 ){
		currentBlock = curb;
	}
	//__threadfence_system();
	__threadfence();
	//__syncthreads();

	int2 resolution = make_int2(width, height);

	// total number of blocks, current block
	do {
		int bx = currentBlock % bmx;
		int by = currentBlock / bmx;

		unsigned int xoffset = bx * blockDim.x;
		unsigned int yoffset = by * blockDim.y;

		unsigned int x = tidx + xoffset;
		unsigned int y = tidy + yoffset;

		if( x > width - 1 || y > height - 1 ) return;

		float3 c;
		float2 offset = generateRandomNumberFromThread2(resolution, time, x, y);

		float u = x + offset.x;
		float v = y + offset.y;

		u = u / (float) width - 0.5;
		v = v / (float) height - 0.5;

		Ray r = generateRay(cam, u, v);

		if( isRayTracing ) {
			c = traceRay_simple(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);
		}
		else c = traceRay_general(time, resolution, x, y, r, inShapesCount, inShapes, inLightsCount, inLights, inMaterialCount, inMaterial);

#define SHOW_AFFINITY 0
#if SHOW_AFFINITY
		c.x = blockIdx.x / (float)gridDim.x;
		c.y = blockIdx.y / (float)gridDim.y;
		c.z = 1.0 - c.x;

		/*
		c.c.r = threadIdx.x / (float)blockDim.x;
		c.c.g = threadIdx.y / (float)blockDim.y;
		c.c.b = 0.0;
		*/
#endif
		//__syncthreads();
		// write output vertex
		color[y*width+x] += c;	
		__syncthreads();
		//__threadfence_block();

		if( tid == 0 ){
			currentBlock = atomicAdd(&curb, 1);
		}
		__threadfence();


	}while(currentBlock < ttlb);
}
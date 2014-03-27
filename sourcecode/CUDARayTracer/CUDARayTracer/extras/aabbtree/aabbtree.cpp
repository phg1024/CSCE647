#include "aabbtree.h"

#include <fstream>
#include <queue>
#include <list>
using std::queue;
using std::list;
using std::ofstream;
using std::endl;

namespace aabbtree {

bool AABB::intersectTest(const float3 &origin, const float3 &dir, const float3 &invDir)
{
	float3 rdirinv = invDir;
	
	rdirinv.x = (dir.x==0)?FLT_MAX:rdirinv.x;
	rdirinv.y = (dir.y==0)?FLT_MAX:rdirinv.y;
	rdirinv.z = (dir.z==0)?FLT_MAX:rdirinv.z;
	
	float tmin, tmax;

	float l1   = (minPt.x - origin.x) * rdirinv.x;
	float l2   = (maxPt.x - origin.x) * rdirinv.x;
	tmin = fminf(l1,l2);
	tmax = fmaxf(l1,l2);

	l1   = (minPt.y - origin.y) * rdirinv.y;
	l2   = (maxPt.y - origin.y) * rdirinv.y;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	l1   = (minPt.z - origin.z) * rdirinv.z;
	l2   = (maxPt.z - origin.z) * rdirinv.z;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	return ((tmax >= tmin) && (tmax >= 0.0f));
}

AABBTree::AABBTree(): Tree<AABBNode>() {
nodeCount[0] = 0; nodeCount[1] = 0; nodeCount[2] = 0;
}

bool AABBTree::intersectTest(const float3 &origin,
                                    const float3 &destination,
                                    float &t, Triangle& tri)
{
    float3 dir = destination - origin;
    float3 invDir = 1.0 / dir;
    if( dir.x == 0 ) invDir.x = numeric_limits<double>::max();
    if( dir.y == 0 ) invDir.y = numeric_limits<double>::max();
    if( dir.z == 0 ) invDir.z = numeric_limits<double>::max();

    bool flag = false;

    // traverse the tree to look for a intersection
    t = numeric_limits<float>::max();

    queue<AABBNode*> nodeset;
    //int testCount = 0;
    nodeset.push(mRoot);
    while(!nodeset.empty())
    {
        AABBNode* node = nodeset.front();
        nodeset.pop();

        if( node->aabb.intersectTest(origin, dir, invDir) )
        {
            if( node->type == AABBNode_Serial::EMPTY_NODE )
                // empty node, ignore it
                continue;

            if( node->type == AABBNode_Serial::INTERNAL_NODE )
            {
                if( node->leftChild != nullptr )
                {
                    //testCount ++;
                    nodeset.push(node->leftChild);
                }
                if( node->rightChild != nullptr )
                {
                    //testCount ++;
                    nodeset.push(node->rightChild);
                }

                continue;
            }
            else
            {
                // hit an AABB
                float tHit;
				/*
                if( node->tri.intersectionTest(origin, destination, tHit) )
                {
                    if( tHit < t )
                    {
                        t = tHit;
                        tri = node->tri;
                        flag = true;
                    }
                }
				*/
            }
        }
    }

    //cout << "testCount = " << testCount << endl;
    return flag;
}

AABBTree::AABBTree(const vector<Triangle>& tris)
{
	memset(nodeCount, 0, sizeof(size_t)*3);
	memset(nodeCountLevel, 0, sizeof(size_t)*128);
	maxDepth = 0;
    cout << "building AABB tree ..." << endl;
#if 1
    mRoot = buildAABBTree(tris);
#else
	mRoot = buildAABBTree_SAH(tris, SplittingPlane());
#endif
    cout << "AABB tree built." << endl;
}

AABBTree::~AABBTree()
{
    releaseTree(mRoot);
}

void AABBTree::releaseTree(AABBNode *node)
{
    if( node!= nullptr )
    {
        releaseTree(node->leftChild);
        releaseTree(node->rightChild);
        delete node->leftChild;
        delete node->rightChild;
    }
}

void AABBTree::printNodeStats()
{
	cout << "Total nodes: " << nodeCount[0] + nodeCount[1] + nodeCount[2] << endl;
	cout << "Empty nodes: " << nodeCount[0] << endl;
	cout << "Internal nodes: " << nodeCount[1] << endl;
	cout << "Leaf nodes: " << nodeCount[2] << endl;
	cout << "Max depth: " << maxDepth << endl;
	for(int i=0;i<maxDepth;i++) {
		cout << "Nodes at level " << i << ": " << nodeCountLevel[i] << endl;
	}
}

vector<AABBNode_Serial> AABBTree::toArray() const {
	cout << "Serializing the AABB tree ..." << endl;
	cout << "reserving " << sizeof(AABBNode_Serial)*(nodeCount[2]*2+1) << " bytes ..." << endl;
	vector<AABBNode_Serial> A;
	A.reserve(nodeCount[2]*2+1);
	cout << "done." << endl;
		
	queue<AABBNode*> nodeset;
	nodeset.push(mRoot);

	size_t totalNode = 0;
	while(!nodeset.empty())
	{
		AABBNode* node = nodeset.front();
		nodeset.pop();

		if( node->type == AABBNode_Serial::INTERNAL_NODE )
		{			
			int leftIdx = -1, rightIdx = -1;
			if( node->leftChild != nullptr )
			{
				nodeset.push(node->leftChild);
				leftIdx = ++totalNode;
			}
			if( node->rightChild != nullptr )
			{
				nodeset.push(node->rightChild);
				rightIdx = ++totalNode;
			}
			A.push_back(node->serialize(leftIdx, rightIdx));
		}
		else if( node->type == AABBNode_Serial::LEAF_NODE )
		{
			// leaf node, simply push it to the array
			A.push_back(node->serialize(-1, -1));
			
			// bug, no need to add this anymore, because the space is already counted before
			//totalNode++;
		}
		else {
			// not suppose to see this
			cout << "empty node." << endl;
		}
	}

#define WRITE_TREE 0
#if WRITE_TREE

	ofstream fout("tree.txt");
	for(int i=0;i<A.size();i++) {
		fout << "idx = " << i << ": " << A[i] << endl;
	}
	fout.close();

#endif

	cout << "Serialized. Total nodes = " << A.size() << endl;
	return A;
}

AABBNode* AABBTree::buildAABBTree(const vector<Triangle>& inTris, int level)
{
	auto tris = inTris;
    // build an AABB Tree from a triangle mesh

    // for all faces in the indices set, compute their centers
    // also get the AABB of these faces

	AABBNode* node = new AABBNode;
	nodeCountLevel[level]++;
	maxDepth = max(level, maxDepth);

    if( tris.empty() )
    {
        return nullptr;
    }    
    else if( tris.size() <= MAX_TRIS_PER_NODE )
    {
		nodeCount[2]++;
		node->type = AABBNode_Serial::LEAF_NODE;
		
		for(int i=0;i<tris.size();i++) {
			node->tri[i] = tris[i];
		}
		node->ntris = tris.size();
		node->aabb = AABB(&(tris[0]), tris.size());
        node->leftChild = nullptr;
        node->rightChild = nullptr;
        return node;
    }
	else {
		nodeCount[1]++;
		node->type = AABBNode_Serial::INTERNAL_NODE;
		node->ntris = 0;
		node->aabb = AABB(&(tris[0]), tris.size());

		if( level == 0 ) {
			// top level bounding box
			cout << node->aabb.minPt.x << ", "
				<< node->aabb.minPt.y << ", "
				<< node->aabb.minPt.z << "; " 
				<< node->aabb.maxPt.x << ", "
				<< node->aabb.maxPt.y << ", "
				<< node->aabb.maxPt.z
				<< endl;
		}

		// sort the pairs along the longest axis
		float3 ranges = node->aabb.range();

		if( ranges.x >= ranges.y && ranges.x >= ranges.z )
		{
			// sort along x axis
			std::sort(tris.begin(), tris.end(), [](const Triangle& t1, const Triangle& t2) -> bool {
				float3 c1 = t1.center();
				float3 c2 = t2.center();
				return c1.x < c2.x;
			});
		}
		else if( ranges.y >= ranges.x && ranges.y >= ranges.z ) {
			// sort along x axis
			std::sort(tris.begin(), tris.end(), [](const Triangle& t1, const Triangle& t2) -> bool {
				float3 c1 = t1.center();
				float3 c2 = t2.center();
				return c1.y < c2.y;
			});
		}
		else {
			// sort along x axis
			std::sort(tris.begin(), tris.end(), [](const Triangle& t1, const Triangle& t2) -> bool{
				float3 c1 = t1.center();
				float3 c2 = t2.center();
				return c1.z< c2.z;
			});
		}

		// split the sorted faces set, build two children nodes
		vector<Triangle> leftSet, rightSet;
		int midPoint = tris.size() / 2;

		//cout << fcpairs.size() << ", " << midPoint << endl;
		leftSet.assign(tris.begin(), tris.begin() + midPoint);
		rightSet.assign(tris.begin() + midPoint, tris.end());

		node->leftChild = buildAABBTree(leftSet, level+1);
		node->rightChild = buildAABBTree(rightSet, level+1);

		return node;
	}
}

// build a tree with surface area heuristics
AABBNode* AABBTree::buildAABBTree_SAH(const vector<Triangle>& inTris, const SplittingPlane& pprev, int level )
{
	// some constants
	const float minCv = 6.0;
	const float Ci = 1.0, Cs = 3.0;

	auto tris = inTris;
	if( tris.empty() ) return nullptr;
	// build an AABB Tree from a triangle mesh

	// for all faces in the indices set, compute their centers
	// also get the AABB of these faces

	AABBNode* node = new AABBNode;
	nodeCountLevel[level]++;
	maxDepth = max(level, maxDepth);

	// best cost for this node
	float Cp;
	SplittingPlane p;
	PlaneSide pside;

	AABB bb(&tris[0], tris.size());
	findBestPlane(tris,  bb, p, Cp, pside);

	if( tris.size() < MAX_TRIS_PER_NODE || Cp < Ci*tris.size() || p == pprev )
	{
		if( tris.size() < MAX_TRIS_PER_NODE ) {
			// create a leaf node
			nodeCount[2]++;
			node->type = AABBNode_Serial::LEAF_NODE;

			for(int i=0;i<tris.size();i++) {
				node->tri[i] = tris[i];
			}
			node->ntris = tris.size();
			node->aabb = AABB(&(tris[0]), tris.size());
			node->leftChild = nullptr;
			node->rightChild = nullptr;
			return node;
		}
		else {
			// build a naive tree instead
			return buildAABBTree(tris, level);
		}
	}
	else {
		AABB bbl, bbr;
		splitVoxel(bb, p, bbl, bbr);
		vector<Triangle> leftSet, rightSet;
		sortTrianglesToVoxel(tris, p, pside, leftSet, rightSet);

		// create an internal node

		nodeCount[1]++;
		node->type = AABBNode_Serial::INTERNAL_NODE;
		node->ntris = 0;
		node->aabb = bb;

		// sort the pairs along the longest axis
		float3 ranges = node->aabb.range();
		
		node->leftChild = buildAABBTree_SAH(leftSet, p, level+1);
		node->rightChild = buildAABBTree_SAH(rightSet, p, level+1);

		return node;
	}
}

void AABBTree::findBestPlane(const vector<Triangle>& T, const AABB& V, SplittingPlane& p_est, float& C_est, PlaneSide& pside_est)
{
	C_est = numeric_limits<float>::max();

	for(int axis=0;axis<3;axis++) {
		vector<SplitEvent> events;
		events.reserve(T.size()*2);

		for(int i=0;i<T.size();++i) {
			auto ti = T[i];
			AABB tibb = V.clip(ti);

			if(tibb.isPlanar()) {
				events.push_back(SplitEvent(ti, axis, tibb.minPos(axis), SplitEvent::LyingOnPlane));
			}
			else {
				events.push_back(SplitEvent(ti, axis, tibb.minPos(axis), SplitEvent::StartingOnPlane));
				events.push_back(SplitEvent(ti, axis, tibb.maxPos(axis), SplitEvent::EndingOnPlane));
			}
		}

		// sort the events
		sort(events.begin(), events.end());

		// test every candidate plane
		int NL = 0, NP = 0, NR = T.size();
		for(int i=0;i<events.size();++i) {
			auto p = events[i].p;
			
			int tLyingOnPlane = 0, tEndingOnPlane=0, tStartingOnPlane=0;
			while( i<events.size() && events[i].p.pe == p.pe && events[i].type == SplitEvent::EndingOnPlane ) {
				++tEndingOnPlane;
				i++;
			}
			while( i<events.size() && events[i].p.pe == p.pe && events[i].type == SplitEvent::LyingOnPlane ) {
				++tLyingOnPlane;
				i++;
			}
			while( i<events.size() && events[i].p.pe == p.pe && events[i].type == SplitEvent::StartingOnPlane ) {
				++tStartingOnPlane;
				i++;
			}

			NP = tLyingOnPlane;
			NR -= tLyingOnPlane;
			NR -= tEndingOnPlane;

			float C;
			PlaneSide ps;
			SAH(p, V, NL, NR, NP, C, ps);

			if(C < C_est) {
				C_est = C;
				p_est = p;
				pside_est = ps;
			}
			NL += tStartingOnPlane;
			NL += tLyingOnPlane;
			NP = 0;
		}
	}
}

void AABBTree::splitVoxel(const AABB& V, const SplittingPlane& p, AABB& VL, AABB& VR)
{
	VL = V;
	VR = V;
	VL.maxPos(p.axis) = p.pe;
	VR.minPos(p.axis) = p.pe;
}

void AABBTree::sortTrianglesToVoxel(const vector<Triangle>& T, const SplittingPlane& p, const PlaneSide& pside, vector<Triangle>& TL, vector<Triangle>& TR)
{
	for(int i=0;i<T.size();++i) {
		AABB tbox(T[i]);

		// the triangle is on the splitting plane exactly
		if(tbox.minPos(p.axis) == p.pe && tbox.maxPos(p.axis) == p.pe) {
			if(pside == PlaneSide::Left)
				TL.push_back(T[i]);
			else if(pside == PlaneSide::Right)
				TR.push_back(T[i]);
		} else {
			if(tbox.minPos(p.axis) < p.pe)
				TL.push_back(T[i]);
			if(tbox.maxPos(p.axis) > p.pe)
				TR.push_back(T[i]);
		}
	}
}

// probability of hitting the subvoxel Vsub given that the voxel V was hit
inline float P_Vsub_given_V(const AABB& Vsub, const AABB& V) {
	float SA_Vsub = Vsub.surfaceArea();
	float SA_V = V.surfaceArea();
	return(SA_Vsub/SA_V);
}

// bias for the cost function s.t. it is reduced if NL or NR becomes zero
inline float lambda(int NL, int NR, float PL, float PR) {
	if((NL == 0 || NR == 0) &&
		!(PL == 1 || PR == 1) // NOT IN PAPER
		)
		return 0.8f;
	return 1.0f;
}

// cost C of a complete tree approximated using the cost CV of subdividing the voxel V with a plane p
inline float C(float PL, float PR, int NL, int NR) {
	const float KT = 1.0;
	const float KI = 2.5;//1.0;
	return(lambda(NL, NR, PL, PR) * (KT + KI * (PL * NL + PR * NR)));
}

void AABBTree::SAH(const SplittingPlane& p, const AABB& V, int NL, int NR, int NP, float& CP, PlaneSide& pside)
{
	CP = numeric_limits<float>::max();
	AABB VL, VR;
	splitVoxel(V, p, VL, VR);
	float PL, PR;
	PL = P_Vsub_given_V(VL, V);
	PR = P_Vsub_given_V(VR, V);
	if(PL == 0 || PR == 0) // NOT IN PAPER
		return;
	if(V.range(p.axis) == 0) // NOT IN PAPER
		return;
	float CPL, CPR;
	CPL = C(PL, PR, NL + NP, NR);
	CPR = C(PL, PR, NL, NP + NR );
	if(CPL < CPR) {
		CP = CPL;
		pside = PlaneSide::Left;
	} else {
		CP = CPR;
		pside = PlaneSide::Right;
	}
}

}
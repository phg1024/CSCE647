#include "aabbtree.h"

#include <queue>
#include <list>
using std::queue;
using std::list;

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
    mRoot = buildAABBTree(tris);
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
	vector<AABBNode_Serial> A;
	A.reserve(nodeCount[2]*2);
		
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
		else
		{
			// empty node, simply push it to the array
			A.push_back(node->serialize(-1, -1));
			totalNode++;
		}
	}

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
        // empty node
		nodeCount[0]++;
        node->type = AABBNode_Serial::EMPTY_NODE;
        node->leftChild = nullptr;
        node->rightChild = nullptr;
        return node;
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

}
#pragma once

template <typename NodeType>
class Tree{
public:
	Tree();
	~Tree();

	const NodeType* root() const
	{
		return mRoot;
	}

private:
	// non-copyable
	Tree(const Tree&);
	Tree& operator=(const Tree&);

protected:
	NodeType *mRoot;
};

template <typename NodeType>
Tree<NodeType>::Tree():
	mRoot(0)
{
}

template <typename NodeType>
Tree<NodeType>::~Tree()
{}
cmake_minimum_required(VERSION 3.9)
project(DecisionTree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

add_executable(DecisionTree main.cpp Tree/TreeNode.h Splitter/SplitInfo.h Global/GlobalConsts.h Tree/NodeStats.h Util/Maths.h Dataset/Dataset.h Tree/TreeParams.h TreeBuilder/ParallelTreeBuilder.cpp TreeBuilder/ParallelTreeBuilder.h Dataset/StoredTree.h Splitter/TreeNodeSplitter.h TreeBuilder/TreeBuilder.cpp TreeBuilder/TreeBuilder.h Predictor/TreePredictor.h TreeBuilder/ForestBuilder.cpp TreeBuilder/ForestBuilder.h Tree/ForestParams.h Test/UnitTest.h Dataset/SubDataset.h Tree/ParallelTreeNode.h Splitter/SplitManipulator.h Splitter/TreeNodeSplitter.cpp)
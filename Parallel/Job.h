
#ifndef DECISIONTREE_JOB_H
#define DECISIONTREE_JOB_H

#include <cstdint>
#include <cstring>
#include "../Global/GlobalConsts.h"
#include "../Generics/TypeDefs.h"
#include "../Tree/TreeNode.h"

class Job {
 public:
  static const uint32_t MinValue = 0;
  static const uint32_t WriteToTree = 1;
  static const uint32_t MakeLeaf = 2;
  static const uint32_t FindSplitOnOneFeature = 3;
  static const uint32_t FindSplitOnAllFeatures = 4;
  static const uint32_t DoSplit = 5;
  static const uint32_t InitSplit = 6;
  static const uint32_t SetupRoot = 7;
  static const uint32_t Idle = UINT32_MAX - 1;
  static const uint32_t MaxValue = UINT32_MAX;

  uint32_t type;
  uint32_t tree_id;
  TreeNode *node;
  uint32_t feature_idx;

  static Job Min() {
    Job job;
    job.type = MinValue;
    job.tree_id = 0;
    job.node = nullptr;
    job.feature_idx = 0;
    return job;
  }

  static Job Max() {
    Job job;
    job.type = MaxValue;
    job.tree_id = 0;
    job.node = nullptr;
    job.feature_idx = 0;
    return job;
  }

  static Job IdleJob() {
    Job job;
    job.type = Idle;
    job.tree_id = 0;
    job.node = nullptr;
    job.feature_idx = 0;
    return job;
  }

  static Job SetupRootJob(uint32_t tree_id) {
    Job job;
    job.type = SetupRoot;
    job.tree_id = tree_id;
    job.node = nullptr;
    job.feature_idx = 0;
    return job;
  }

  static Job InitSplitJob(TreeNode *node) {
    Job job;
    job.type = InitSplit;
    job.tree_id = 0;
    job.node = node;
    job.feature_idx = 0;
    return job;
  }

  static Job FindSplitOnOneFeatureJob(TreeNode *node,
                                      uint32_t feature_idx) {
    Job job;
    job.type = FindSplitOnOneFeature;
    job.tree_id = 0;
    job.node = node;
    job.feature_idx = feature_idx;
    return job;
  }

  static Job DoSplitJob(TreeNode *node) {
    Job job;
    job.type = DoSplit;
    job.tree_id = 0;
    job.node = node;
    job.feature_idx = 0;
    return job;
  }

  static Job WriteToTreeJob(uint32_t tree_id) {
    Job job;
    job.type = WriteToTree;
    job.tree_id = tree_id;
    job.node = nullptr;
    job.feature_idx = 0;
    return job;
  }

  Job(const Job &job) {
    memcpy(this, &job, sizeof(Job));
  }

  Job &operator=(const Job &job) {
    memcpy(this, &job, sizeof(Job));
    return *this;
  }

  bool operator<(const Job &job) const {
    if (this->type != job.type) {
      return this->type < job.type;
    } else if (this->tree_id != job.tree_id) {
      return this->tree_id < job.tree_id;
    } else if (this->feature_idx != job.feature_idx) {
      return this->feature_idx < job.feature_idx;
    } else if (this->type == DoSplit) {
      return this->node->Split()->gain > job.node->Split()->gain;
    } else {
      return this->node < job.node;
    }
  }

  bool operator>(const Job &job) const {
    return !operator<(job) && !operator==(job);
  }

  bool operator==(const Job &job) const {
    return type == job.type && tree_id == job.tree_id && node == job.node && feature_idx == job.feature_idx;
  }

 private:
  Job() = default;
};

#endif

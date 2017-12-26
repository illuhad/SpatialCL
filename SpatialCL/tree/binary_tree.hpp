/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2017 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef BINARY_TREE_HPP
#define BINARY_TREE_HPP

#include "../configuration.hpp"
#include "../bit_manipulation.hpp"


/**
General binary tree layout
====================

Nodes for each level in the tree are laid out in continous order in global memory
(this helps coalescing). The leaf level comes first, followed the first level of parent
nodes, and so on, until the root, which is the last node in memory.

For the indexing of nodes it is assumed that the number of leaves is always a number
of 2, as it leads to convenient binary representations of node adresses/offsets
(see below). The number of leaves can be non-powers of 2, in which case they will be
rounded to the nearest power of 2 (=effective_num_leaves).
It is not necessary that all the "padded" leaves actually exist in memory, as client code
can just add (num_leaves-effective_num_leaves) to the indices to correct for this offset.

For the indexing, levels are counted from the root (= Level 0) to the leaves.

Example node layout for 32 particles (Offset is the index where the nodes for a given
level would start in memory):
N_nodes     Level Offset
100000 = 32 l5    000000
010000 = 16 l4    100000
001000 = 8  l3    110000
000100 = 4  l2    111000
000010 = 2  l1    111100
000001 = 1  l0    111110

This means that the level of a node and its position within the level can be identified
by the number of leading 1s of its index.

Terminology
============

* A global index/id is the overall index of a node, counted from the first leaf.
  If the number of leaves is a power of 2, a node's data can simply be accessed
  by reading the node array at the node's global index (If the number of leaves
  is not a power 2, the global index first needs to be corrected by the deviation
  from the power of 2, see above)
* The level offset is the global index of the first node of a given level.
* A local index/id is the index of a node relative to the level offset.
(More intuitively, the local index is obtained by enumerating the nodes within
one node level, while the global index is obtained by enumerating all nodes
across all levels)

The corresponding tree would look like this:
| x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |
|   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |   x   |
|       x       |       x       |       x       |       x       |       x       |       x       |       x       |       x       |
|               x               |               x               |               x               |               x               |
|                                                               x                                                               |

What if the number of leaves is not a power of 2?
=================================================

Treating the case of non power of 2 numer of leaves may require additional effort,
depending on the tree algorithm that one wishes to run. Let's consider the example
of 6 leaves. For the indexing, this will be rounded to the nearest power of 2,
i.e., 8. Hence, the effective tree will look like this (x denotes populated nodes):

| x | x | x | x | x | x |   |   |
|   x   |   x   |   x   |       |
|       x       |       x       |
|               x               |

As can be seen, non-power-of 2 numbers of leaves in general also imply that some
nodes of the higher levels are invalid. It is important when constructing the tree
and also for client code evaluating the tree (in whatever way this may be) to make sure
these invalid nodes are ignored.
The validity of a node can be checked by making sure that the index of the first leaf
contained by a node is smaller than the number of leaves. The index of the first leaf
can be easily obtained using binary_tree_get_leaves_begin(). A shortcut for this check
exists in the form of the binary_tree_is_node_used() function.

*/


namespace spatialcl {

class binary_tree
{
public:
  QCL_MAKE_MODULE(binary_tree)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(bit_manipulation)
    R"(
      #define BT_EFFECTIVE_NUM_LEAVES(num_particles) get_next_power_of_two(num_particles)
      #define BT_LEVEL_OFFSET_MASK(num_levels) n_bits_set(num_levels)
      #define BT_LEVEL_OFFSET(level, num_levels) (~n_bits_set(level+1) & BT_LEVEL_OFFSET_MASK(num_levels))
      #define BT_LEAVES_PER_NODE(level, num_levels) (1ul << (num_levels - level - 1))
      #define BT_NUM_NODES(level) (1ul << level)
    )"
    QCL_RAW
    (

      typedef ulong index_type;

      typedef struct
      {
        uint level;
        index_type local_node_id;
      } binary_tree_key_t;


      void binary_tree_key_init(binary_tree_key_t* ctx,
                                uint level,
                                index_type local_node_id)
      {
        ctx->level = level;
        ctx->local_node_id = local_node_id;
      }

      index_type binary_tree_key_encode_global_id(binary_tree_key_t* ctx,
                                               index_type num_levels)
      {
        index_type offset = BT_LEVEL_OFFSET(ctx->level, num_levels);
        return offset + ctx->local_node_id;
      }

      void binary_tree_key_decode_global_id(binary_tree_key_t* ctx,
                                            index_type global_node_id,
                                            index_type level,
                                            index_type num_levels)
      {
        ctx->level = level;

        index_type offset = BT_LEVEL_OFFSET(level, num_levels);
        ctx->local_node_id = global_node_id - offset;
      }

      index_type binary_tree_get_leaves_begin(binary_tree_key_t* node_key,
                                             index_type num_levels)
      {
        index_type leaves_per_node = BT_LEAVES_PER_NODE(node_key->level, num_levels);

        return node_key->local_node_id * leaves_per_node;
      }

      index_type binary_tree_get_leaves_end(binary_tree_key_t* node_key,
                                           index_type num_levels)
      {
        index_type BT_LEAVES_PER_NODE = BT_LEAVES_PER_NODE(node_key->level, num_levels);

        return (node_key->local_node_id + 1) * BT_LEAVES_PER_NODE;
      }

      int binary_tree_is_node_used(binary_tree_key_t* node_key,
                                   index_type num_levels,
                                   index_type num_leaves)
      {
        return binary_tree_get_leaves_begin(node_key, num_levels) < num_leaves;
      }

      binary_tree_key_t binary_tree_get_children_begin(binary_tree_key_t* node)
      {
        binary_tree_key_t child = *node;

        child.level++;
        child.local_node_id <<= 1;

        return child;
      }

      binary_tree_key_t binary_tree_get_children_last(binary_tree_key_t* node)
      {
        binary_tree_key_t child = binary_tree_get_children_begin(node);
        child.local_node_id++;

        return child;
      }

      binary_tree_key_t binary_tree_get_parent(binary_tree_key_t* ctx)
      {
        binary_tree_key_t result = *ctx;
        result.level--;
        result.local_node_id >>= 1;

        return result;
      }

      int binary_tree_is_left_child(binary_tree_key_t* ctx)
      {
        return (ctx->local_node_id & 1) == 0;
      }

      int binary_tree_is_right_child(binary_tree_key_t* ctx)
      {
        return ctx->local_node_id & 1;
      }
    )
  )
};

}

#endif


# AGraphlets
Approximate Graphlets Extraction 


## Implementation details

### Graph
Input format: tab separated weighted edgelist (nodes represented with integer ids).

```
src			trg			w
node_id0    node_id1	weight
```

### Node Attributes

Input format: tab separated nodelist (nodes represented with integer ids).

```
node_id0    nlabel
```

# Execution

The algorithm can be used as standalone program as well as integrated in python scripts.

## Standalone

```bash

python AGraphlets.py network_filename percentile backbone_threshold min_graphlet_size max_graphlet_size -a node_attr_file
```

where:
* network_filename: edgelist filename
* percentile: component filtering percentile in [0, 100]
* backbone_threshold: edge statistical significance threshold in [0,1]
* min_graphlet_size: minimum size for graphlet patterns
* max_graphlet_size: maximum size for graphlet patterns 
* node_attr_file: node labels filename (optional)

## As python library

```python
import AGraphlets as a
ag = a.AGraphlet(network_file, node_attr=node_attr, approx_percentile=percentile, backbone_threshold=backbone_threshold)
ag.execute(min_pattern_size=min_graphlet_size, max_pattern_size=max_graphlet_size)
```

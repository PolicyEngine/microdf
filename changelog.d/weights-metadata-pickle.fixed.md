Weights and the weight-column name now survive pickling and
to_pickle/read_pickle. Weighted aggregations also remain available on
unpickled MicroDataFrames, and weights are preserved by copy.deepcopy.
Renaming an object preserves its copied weights instead of sharing mutable
weight state with the original object.

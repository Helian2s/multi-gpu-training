# EXP-12 Hypothesis

Pipeline parallel efficiency depends more on stage balance and microbatch count
than on the mere fact that a model is split across two GPUs.

For the AWS-A2 adaptation, the expected result is that increasing the number of
microbatches lowers the estimated two-stage bubble fraction, while an imbalanced
stage split makes the slower rank set the step time. The run records that this
is a PCIe single-node environment and does not claim NVLink behavior.

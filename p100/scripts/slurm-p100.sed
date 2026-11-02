s/^NodeName=slerm-node03 /# slerm-node03 retired 2026-10-04: its GTX 1660 SUPER is gone and both Tesla P100s are in node04.\n#NodeName=slerm-node03 /
/^NodeName=slerm-node04 /s/ CPUs=6 / CPUs=12 /
/^NodeName=slerm-node04 /s/ CoresPerSocket=6 / CoresPerSocket=12 /
/^NodeName=slerm-node04 /s/ RealMemory=9614 / RealMemory=15600 /
/^NodeName=slerm-node04 /s/gpu:pascal:1/gpu:pascal:2/
s/^\(PartitionName=\(gpu\|llm\) Nodes=\)slerm-node\[03-04\]/\1slerm-node04/

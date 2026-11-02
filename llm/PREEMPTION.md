# Optional: let batch GPU jobs reclaim a card from the persistent server

Without this, batch and endpoint still coexist — Slurm will not hand the same
GPU to two jobs, so a batch job simply takes the *other* card, and queues if
both are busy. This config is only needed if you want a batch job to be able to
evict a running server rather than wait for it.

`PreemptType` is currently null and `PreemptMode=OFF`, so nothing preempts today.

## The change

In `/apps/slurm.conf`:

1. Add to the scheduler section (near `SchedulerType=sched/backfill`):

       PreemptType=preempt/partition_prio
       PreemptMode=REQUEUE

2. Replace line 48 (the `gpu` partition) and add an `llm` partition beside it:

       PartitionName=gpu Nodes=slerm-node[03-04] MaxTime=INFINITE State=UP DefMemPerCPU=1500 PriorityTier=100
       PartitionName=llm Nodes=slerm-node[03-04] MaxTime=INFINITE State=UP DefMemPerCPU=1500 PriorityTier=1 PreemptMode=REQUEUE GraceTime=30

The two partitions overlap the same two nodes on purpose. `gpu` (tier 100)
outranks `llm` (tier 1), so a job submitted to `gpu` preempts a server sitting
in `llm`. `REQUEUE` means the server job goes back to the queue and restarts by
itself once the batch job releases the card — no babysitting. `GraceTime=30`
gives llama-server 30s to shut down on SIGTERM before it is killed.

## Deploying it

slurm.conf must be byte-identical on the head and all four nodes, and the nodes
are not reachable from the head — run the copy from the desktop.

    # on slerm-head, as root: edit the master copy
    sudoedit /apps/slurm.conf

    # from the DESKTOP, push it to every host and reload
    for h in slerm-head slerm-node01 slerm-node02 slerm-node03 slerm-node04; do
      ssh $h "sudo cp /apps/slurm.conf /etc/slurm/slurm.conf"
    done
    ssh slerm-head "sudo systemctl restart slurmctld"
    for h in slerm-node01 slerm-node02 slerm-node03 slerm-node04; do
      ssh $h "sudo systemctl restart slurmd"
    done
    ssh slerm-head "scontrol show config | grep -i preempt"

Adding a partition needs a real `slurmctld` restart; `scontrol reconfigure` is
not enough.

## After it is in

Submit the persistent server to `llm` instead of `gpu`:

    sbatch --partition=llm --gres=gpu:turing:1 /data/llm/jobs/server.sh

Batch jobs keep using `--partition=gpu` (the default in `jobs/batch.sh`) and
will now evict the server when both cards are busy.

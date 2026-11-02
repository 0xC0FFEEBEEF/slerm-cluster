# slerm

A Slurm cluster I built at home to learn Slurm end to end. Slurm is compiled from source; the cluster has a
dedicated controller (`slerm-head`), NFS storage every node mounts, MUNGE auth, cgroup-enforced scheduling, CPU
and GPU partitions and job accounting. The nodes are VMs on two Proxmox servers plus one bare-metal machine
(`deadlock`).

Everything the cluster does is a Slurm job:

- **NOAA radar**: NEXRAD Level 2 and HRRR data from NOAA's open data, one array task per scan, rendered into
  GIFs for a wall display.
- **Wazuh analysis**: alerts pulled from Wazuh indexers into a snapshot, counted in a sharded array job, then a
  dependent reduce job that scores the whole window and writes a report, with optional AI triage.
- **Builds**: llama.cpp builds and tests for the GPU node, CPU builds and AppImages on the compute nodes.
- **LLM serving**: one OpenAI-compatible gateway on the head. A request brings up the nodes, workers and
  llama.cpp router the model needs, waits for them, then answers; idle models are unloaded and the nodes power
  off again.

More about the lab around it: [nbdevlab.com/homelab](https://www.nbdevlab.com/homelab).

| Folder | Lives at | What it is |
|---|---|---|
| `panel/` | `slerm-head:~/slerm-panel` | Control panel on port 8080: submit, schedule and watch jobs, frame viewer and GIF builder, the Wazuh pipeline, GPU and LLM usage. See `panel/README.md`. |
| `orchestrator/` | `slerm-head:~/slerm-orch` | `orch.py`: the gateway on port 8090. It holds a request while it powers on nodes (Proxmox API, BMC), starts workers and the router, then releases idle resources. `ups-push-recv` takes UPS state from the power setup, so the cluster can react to power events. |
| `llm/` | `slerm-head:/data/llm` (NFS, all nodes) | `bin/` helpers, `etc/models.ini` presets and model fetch lists, `jobs/` Slurm scripts for the router, RPC workers and builds. See `llm/README.md`. |
| `p100/` | `slerm-head:/data/llm/src` | Own llama.cpp builds for two Tesla P100s (Pascal). `llama.cpp-v050-p100.patch` is the diff against upstream, `new-files/` the files the patch cannot carry, `local-edits/` the scripts that applied each round of kernel edits, `local-files/` earlier hand-written kernels, `scripts/` load, thermal and gateway tests. |
| `proxmox/` | the Proxmox hosts | VM hookscripts (`slerm-yield.sh`, `slerm-numa-pin.sh`) and the timer that keeps the hookscript attached, so the cluster gives way to other VMs. |
| `nodes/` | compute nodes | Local model cache setup. |
| `tools/` | desktop | `slerm-llm` and `slerm-llm-ctl` command-line clients. |
| `docs/` | | Benchmarks and speed notes. |

## About this copy

This is a cleaned public snapshot of my private working repo, which is what the servers deploy from. Addresses
are replaced with host names, the cluster user is shown as `slerm`, and a few site-specific details and notes
are left out. Secrets and state were never in the repo: Proxmox tokens, BMC and indexer passwords, the panel
database, logs, model files and build directories all stay on the servers.

## License

GNU General Public License v3.0 or later, see [`LICENSE`](LICENSE). The `p100/` patch and kernels build on
[llama.cpp](https://github.com/ggml-org/llama.cpp), which is MIT-licensed by the ggml authors.

<sub>Preparing this public copy from my private repo was done with help from Claude.</sub>

Berzelius GPU User Guide
CUDA
SLURM
Interactive Sessions
Submitting Batch Jobs
NSC boost-tools
NVIDIA Multi-Instance GPU MIG
Multi-node Jobs
GPU Reservations
Resource Allocations Costs
GPU Usage Efficiency Policy
Running Multiple Tasks Concurrently within A Single Job
CUDA
CUDA Driver Version: nvidia-smi reports the version of the NVIDIA GPU driver that is installed on your system. The GPU driver is responsible for enabling GPU hardware and managing GPU resources. The driver version may not necessarily match the version of the CUDA Toolkit.

CUDA Toolkit Version: nvcc -V reports the version of the CUDA compiler (part of the CUDA Toolkit) that is installed on your system. This version indicates the CUDA development environment that is available for compiling and running CUDA programs.

The current CUDA driver on the compute nodes is from the CUDA 12.0 release with compatibility packages for 12.2.

Thu Sep 19 12:16:43 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06             Driver Version: 535.183.06   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:47:00.0 Off |                    0 |
| N/A   30C    P0              56W / 400W |      0MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
There are various CUDA Toolkit installations on Berzelius. Please check via module avail.

SLURM
SLURM (Simple Linux Utility for Resource Management) is an open-source, highly configurable, and widely used workload manager and job scheduler for high-performance computing (HPC) clusters.

When allocating a partial node, NSC’s recommendation is to allocate tasks (CPU cores) in proportion to how large a fraction of the node’s total GPUs you allocate, i.e. for every GPU (1/8 of a node) you allocate, you should also allocate 16 tasks (1/8 = 16/128). The default memory allocation follows that of the cores, i.e. for every CPU core allocated, 7995 MB of RAM (a small bit less than 1/128th of node’s total RAM) is allocated. This is automatically taken care of when using only the switch --gpus=X as in the examples of the quick start guide. On “fat” nodes twice the amount of memory is allocated if the job used the -C "fat" feature flag.

Please note that when allocating more GPUs than 8 (more than one node), you will end up on several nodes and will require multi-node capabilities on your software setup to make use of all allocated GPUs.

Be aware that there are very many ways to give conflicting directives when allocating resources, and they will result in either not getting the allocation or getting the wrong allocation. Before submitting a resource intensive batch job, it’s worthwhile to check out the settings in an interactive session verifying the resources are properly allocated.

Interactive Sessions
An interactive session allows you to work directly on the cluster, interact with the compute nodes, and run commands in a real-time, interactive manner. Interactive sessions are useful for tasks like code development, testing, debugging, and exploring data.

Request 1 GPU with defaults: 16 CPU cores, 128 GB RAM, and a default wall time of 2h.
[user@berzelius001 ~]$ interactive --gpus=1
Request 1 GPU on a specific node.
[user@berzelius001 ~]$ interactive --gpus=1 --nodelist=node045
Request 1 GPU using a specific account when having multiple projects. You can use projinfo to check your project accounts.
[user@berzelius001 ~]$ interactive --gpus=1 -A <your-project-account>
Request 2 GPU for 30 minutes: 32 CPU cores, 256 GB RAM. The time limit format is days-hours:minutes:seconds.
[user@berzelius001 ~]$ interactive --gpus=2 -t 00-00:30:00
Request 1 node (8 GPU) for 6 hours.
[user@berzelius001 ~]$ interactive -N 1 --exclusive -t 6:00:00
Request 1 GPU from a fat node for 30 minutes.
[user@berzelius001 ~]$ interactive --gpus=1 -C "fat" -t 30
Using the feature flag -C "fat" only “fat” nodes will be considered for the job. The corresponding flag for “thin” nodes is -C "thin. If the feature flag is not specified any node can be used. When the “fat” feature is used the job is assigned 254GB of system memory per GPU. The GPUh cost of a “fat” GPU is twice that of a “thin” GPU, even if the job was not launched with -C "fat".

Submitting Batch Jobs
In the context of HPC clusters, batch jobs are computational tasks that are submitted to a job scheduler for execution. Batch job submission is a common way to efficiently manage and execute a large number of computational tasks on HPC systems.

You need to create a batch job script batch_script.sh. Here is an example.

#!/bin/bash
#SBATCH -A <your-project-account>
#SBATCH --gpus 4
#SBATCH -t 3-00:00:00

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch_2.0.1
python some_AI_model.py
You can submit the batch job by:

[user@berzelius001 ~]$ sbatch batch_script.sh
A more detailed introduction of batch jobs can be found here. The wall time limit has been set to 3 days (72h) to ensure that there is reasonable turnover of jobs on Berzelius.

NSC boost-tools
We have NSC boost-tools to add more flexibility to the job scheduling. We currently provide three tools:

nsc-boost-priority: Increase the priority of a job.
nsc-boost-timelimit: Extend the time limit of a job.
nsc-boost-reservation: Reserve nodes for a specific time period.
NVIDIA Multi-Instance GPU (MIG)
NVIDIA Multi-Instance GPU (MIG) is a feature which allows a single GPU to be partitioned into multiple smaller GPU instances, each of which can be allocated to different tasks or users. This technology helps improve GPU utilization and resource allocation in multi-user and multi-workload environments.

Nodes in the reservation 1g.10gb have MIG feature enabled. Each 1g.10gb-instance is equipped with

1/7th of the A100s’ compute capabilities,
10GB VRAM,
2 cores / 4 threads,
32GB RAM.
If your job require more resources than the fix amount you should not use this reservation.

[user@berzelius001 ~]$ interactive --reservation=1g.10gb
Multi-node Jobs
Multi-node jobs for regular MPI-parallel applications should be pretty standard on the cluster, and can use the common mpirun (or mpiexec) launcher available via the buildenv-gcccuda/11.4-8.3.1-bare module or srun --mpi=X (supported X by SLURM are pmi2, pmix and pmix_v3). If your application has been built with NSC provided toolchain(s) you should also be able to launch it with mpprun in standard NSC fashion.

Multi-node jobs using GPUs can be challenging when running in a Apptainer container. For NVIDIA NGC containers used with Apptainer, you can possibly launch your job using the mpirun provided by loading the buildenv-gcccuda/11.4-8.3.1-bare module. See Apptainer’s documentation.

We have a few examples of multi-node jobs available for your reference.

GPU Reservations
Please refer to the NSC boost-tools for how to reserve GPUs/nodes for a specific time period.

Resource Allocations Costs
Depending on the type of resources allocated to a job the cost in GPUh will vary. Using feature flags it is possible to select either “thin” (A100 40GB) or “fat” (A100 80GB) for a job, a job not specifying either can use either. MIG GPUs are accessed through the MIG reservation.

GPU	Internal SLURM cost	GPUh Cost per hour	Accessed through
MIG 1g.10gb	4	0.25	–reservation=1g.10gb
A100 40GB	16	1	-C “thin”, or no flag
A100 80GB	32	2	-C “fat”, or no flag
GPU Usage Efficiency Policy
As the demand for time on Berzelius is high, we need to ensure allocated time is efficiently used. The efficiency of running jobs is monitored continuously by automated systems. Users can, and should, monitor how well jobs are performing. Users working via ThinLinc can use the tool jobgraph to check how individual jobs are performing. For example, jobgraph -j 12345678 will generate a png-file with data on how job 12345678 is running. Please note that for jobarrays you need to look at the “raw” jobid. It is also possible to log onto a node running a job using jobsh -j $jobid and then use tools such as nvidia-smi or nvtop to look at statistics in more detail.

In normal use, we expect jobs properly utilizing the GPUs to pull 200W, with most AI/ML-workloads at above 300W. The average idle power for the GPUs are about 50 to 60 Watt.

Particularly inefficient jobs will be automatically terminated. The following criteria are used to determine if a job should be canceled:

Exponential moving average power utilization per GPU is below 100W. In normal use, we expect jobs properly utilizing the GPUs to pull 200W, with most AI/ML workloads at above 300W.
The job is scheduled without any GPU.
Exceptions from this are:

Jobs that have not yet run for one hour, to allow for some preprocessing at the beginning of jobs,
Interactive jobs for the first 8 hours, started with the NSC interactive tool,
Jobs running within reservations, including devel and safe,
Jobs running on the MIG node
Explicitly whitelisted projects (please contact us if you believe this applies to you).
These criteria are slightly simplified and will be made stricter over time (e.g. interactive jobs will become eligable for termination after a several hour long window). Users are informed about canceled jobs hourly, to avoid sending out too much spam for users getting jobarrays canceled.

Running Multiple Tasks Concurrently within A Single Job
Sometimes, we wish to run larger number of tasks that individually won’t saturate the GPU properly often indicated by low power usage. This is a method for increasing throughput and increase efficiency by running several such tasks concurrently. While each individual task may take a bit longer this way, we can do more work with a given amount of GPU-time.

In this example, we have 24 words we’d like to generate poems based on, using a large language model.

Note that this is only a fun example, there are far better ways to do LLM-inference. Certain types of Alphafold-tasks, as well as training with very small minibatch-sizes are examples where an approach like this can make sense.

The method we describe here can also be beneficial for jobs that benefit from using node-local scratch storage, any data transfers can be amortized over a longer time. In that case, we may run without concurrency, but run many tasks after each other within the same Slurm job.

Make sure to benchmark any specific workload to determine the best way to run it.

First, we create a file with a “batch” of tasks to work on. In this case, we have a file with 24 words, one per line.

[user@berzelius1 xargs-example]$ cat data.txt | wc -l
24

[user@berzelius1 xargs-example]$ head -n 3 data.txt
anniversary
annotated
annotation
Our main workload is the script poet.sh. This scripts will generate a poem based on its first argument and print it to standard out. Errors will end up on stderr.

[user@berzelius1 xargs-example]$ srun --gpus=1 ./poet.sh "hello" 2>/dev/null
Hello,
I'm a simple greeting
[...]
To make sure we store our results, we’d like to use this script from a wrapper, poet_wrapper.sh. The wrapper calls poet.sh and then stores the resulting poem in a text file on disk. In general, using a wrapper like this may or may not be needed depending on the workload. In many cases, one may want to add additional arguments or redirect the output. Using a wrapper is usually a convenient way to do so.

[user@berzelius1 xargs-example]$ cat poet_wrapper.sh
#!/bin/bash
mkdir -p results               # Ensure we create a directory for the results

./poet.sh $1 > results/$1.txt  # Run `./poet.sh` with correct arguments and
                               # redirect the output to a file.
We can now put everything together into a batch script we can launch via sbatch concurrent_poet.sh.

[user@berzelius1 xargs-example]$ cat concurrent_poet.sh
#!/bin/bash
#
#SBATCH -J concurrent_poet
#SBATCH --gpus=1

CONCURRENT_TASKS=4

cat data.txt \              # Read our input data
  | xargs \                 # We use the xargs-utility to launch tasks
    -d '\n' \               # Our input data newline-delimited
    -I {} \                 # Use '{}' as a placeholder for our input data
    -P $CONCURRENT_TASKS \  # Launch $CONCURRENT_TASKS tasks at the same time
    ./poet_wrapper.sh "{}"  # This is the command `xargs` will run. Note the `{}`.
The exact parameters will depend greatly on the workload. Make sure to benchmark different number of concurrent tasks to determine a good value. In this example we are using a simple text file with our input data. If the intention is to run more tasks than fit within 72 hours on a single GPU, the simple data.txt-file may need to be split up into several such files, one for each Slurm job. For long running Slurm jobs, it may be a good idea to add logic that allows for restarts, e.g. don’t run computations when an outdata-file already exists.

After the Slurm job has concluded, we have all our output data available:

[user@berzelius1 xargs-example]$ ls results/ | wc -l
24

[user@berzelius1 xargs-example]$ ls results/ | head -n 3
allergy.txt
anniversary.txt
annotated.txt

[user@berzelius1 xargs-example]$ cat results/annotated.txt  | head -n5
And as I go through life, with each passing year,
I find that there's so much more to learn and share.
Each memory becomes an annotation,
A marking of my thoughts and emotions,
As I explore and grow along the way.
The workflow we describe here is very generic, details must be adapted to specific workloads. If you have any questions regarding this, please contact us!

All code above assume we use a single GPU, but for some cases it may be beneficial to use an exclusive node (for example, if one has a large dataset and want to use all of the available local scratch space). In such cases, you may need to add logic for load-balancing between the available GPUs.
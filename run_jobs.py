# import os
# from sac_experiment import run_experiment

# methods = ["attention_redistribution", "valence_biased_sampling", "optimistic_q", "advantage_reward_shaping"]
# omegas = [
#     [0.01, 0.03, 0.1, 0.3, 1, 3, 10],  # attention redistribution
#     [0.01, 0.03, 0.1, 0.3, 1, 3, 10], # optimistic q
#     [0.01, 0.03, 0.1, 0.3, 1, 3, 10], # valence biased sampling
#     [0.01, 0.03, 0.1, 0.3, 1, 3, 10] # advantage reward shaping
# ]

# # get 200 more trials for each
# batch_size = 1
# num_batches = 10

# for batch in range(num_batches):
#     for i, method in enumerate(methods):
#             for j, omega in enumerate(omegas[i]):
#                 os.system(f"sbatch --output='jobs/outputs/%j_{method}_{omega}.out' jobs/scripts/SAC_job.sh {method} {omega} {batch_size}")
        
#     os.system(f"sbatch --output='jobs/outputs/%j_control.out' jobs/scripts/SAC_job.sh control 0 {batch_size}")

import os
        
for i in range(30):
    os.system('sbatch jobs/scripts/pendulum_array_job.sh')

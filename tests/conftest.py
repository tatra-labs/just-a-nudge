import torch

# The tests use tiny models; one thread avoids OpenMP oversubscription stalls on busy machines
# (seen on Windows: a 32-sequence forward took 11 s with 4 threads and 0.07 s with 1).
torch.set_num_threads(1)

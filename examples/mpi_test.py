from mpi4py import MPI

def run(env):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env.a[rank] = rank
    return 
    


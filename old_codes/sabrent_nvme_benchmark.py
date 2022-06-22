def func(arr, N = 10,M = 10,disk=1):
     global dtlst
     import h5py
     from time import time,sleep
     from ubcs_auxiliary.save_load_object import save_hdf5
     print(f'/mnt/data/test/')
     for i in range(100):
          T = get_T()
          dtlst.append([time(),0,T[0],T[1]])
          sleep(0.1)
     for i in range(N):
         t1 = time()
         for j in range(M):
             for k in range(3):
               with h5py.File('/mnt/data/test/test_data_file.hdf5','a') as f:
                 f.create_dataset(f'frame_{time()}_{i}_{j}_{k}',data = arr)
                
         t2 = time()
         dt = t2-t1
         T = get_T()
         dtlst.append([time(),dt,T[0],T[1]])
         if dt < 1: sleep(1-dt)


def test_thread(arr,N,M,disk):
    print('N = ',N)
    print('M = ',M)
    t1 = time();
    func(arr,N,M,disk);
    t2 = time();
    size = N*M*(3001*4096*2/1024/1024);
    dt = (t2-t1)/(N);
    print(t2-t1,size, dt, size /(t2-t1))


def test(N = 10, M = 19*3, disk = 1):
    from ubcs_auxiliary.threading import new_thread
    arr = (random.rand(1,3001,4096)*4096).astype('int16')
    new_thread(test_thread,arr,N,M,disk)

def get_T():
    import subprocess
    output1 = subprocess.check_output("sudo nvme smart-log /dev/nvme0n1 | grep '^temperature'", shell=True)
    output2 = subprocess.check_output("sudo nvme smart-log /dev/nvme1n1 | grep '^temperature'", shell=True)
    return (int(str(output1).split('C')[0].split(':')[1]),int(str(output2).split('C')[0].split(':')[1]))

def plot():
    from matplotlib import pyplot as plt

    dtarr = array(dtlst)
    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(2, 1, hspace=0.025, wspace=0.025)
    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[1,0])
    ax0.plot(dtarr[:,0],dtarr[:,1])
    ax1.plot(dtarr[:,0],dtarr[:,2])
    ax1.plot(dtarr[:,0],dtarr[:,3])
    plt.show()

N = 100
M = 19*3

dtlst = []

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from numpy import random, array
from time import time, sleep
from ubcs_auxiliary.threading import new_thread

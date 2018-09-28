import torch
import torch.multiprocessing as mp


def foo(worker, tl):
    tl0 = tl[0]
    tl1 = tl[1]
    tl[worker] += (worker + 1) * 1000


if __name__ == '__main__':
    #     mp.set_start_method('spawn')
    Num = 1024
    tl = []
    for i in range(Num):
        tl.append(torch.randn(2))
    #         tl.append(torch.randn(i+1))

    for t in tl:
        t.share_memory_()

    print("before mp: tl=")
    print(tl)

    processes = []
    for i in range(Num):
        p = mp.Process(target=foo, args=(i, tl))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("after mp: tl=")
    print(tl)
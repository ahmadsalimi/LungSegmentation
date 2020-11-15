from threading import Thread, Lock
import numpy as np


class Worker(Thread):

    def __init__(self):
        super(Worker, self).__init__()
        self.coordination_lock = Lock()
        self.coordination_lock2 = Lock()
        self.func = None
        self.args_list = None
        self.results = None
        self.finish = False

    def do_process(self, func, args_list):
        self.func = func
        self.args_list = args_list
        self.results = None
        self.coordination_lock2.acquire()
        self.coordination_lock.release()

    def run(self):
        while not self.finish:
            self.coordination_lock.acquire()

            if self.finish:
                break

            self.results = []

            for x in self.args_list:
                self.results.append(self.func(*x))

            self.coordination_lock.release()

            self.coordination_lock2.acquire()
            self.coordination_lock2.release()

    def get_result(self):
        self.coordination_lock.acquire()
        self.coordination_lock2.release()
        return self.results

    def term(self):
        self.finish = True
        self.coordination_lock.release()


class WorkersCoordinator:

    def __init__(self, n_threads):
        # Making threads
        self.n_threads = n_threads
        self.threads = [Worker() for _ in range(n_threads)]

        # locking threads not to run
        for th in self.threads:
            th.coordination_lock.acquire()

        # Running the threads
        for th in self.threads:
            th.start()

    def run_func(self, func, args_list):

        share = 1.0 * len(args_list) / self.n_threads

        # Separating arguments for each thread!
        for i in range(self.n_threads):
            self.threads[i].do_process(func, args_list[int(np.floor(i * share)):int(np.floor((i + 1) * share))])

        # waiting for them to join
        res = []

        for th in self.threads:
            res += th.get_result()

        return res

    def finish(self):
        for th in self.threads:
            th.term()
            th.join()


''' TEST
if __name__ == '__main__':
    wc = WorkersCoordinator(6)
    res = wc.run_func(lambda x, y: 10 * x + y, [(i, j) for i in range(5) for j in range(5)])
    res = [[res[i * 5 + j] for j in range(5)] for i in range(5)]
    print(res)
    print('Done1')

    res = wc.run_func(lambda x, y: 10 * x + y, [(i, j) for i in range(5) for j in range(5)])
    res = [[res[i * 5 + j] for j in range(5)] for i in range(5)]
    print(res)
    print('Done2')
'''
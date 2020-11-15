from threading import Thread
import numpy as np


class Worker(Thread):

    def __init__(self, func, args_list):
        super(Worker, self).__init__()
        self.func = func
        self.args_list = args_list
        self.results = None

    def run(self):
        self.results = []
        for x in self.args_list:
            self.results.append(self.func(* x))


class WorkersCoordinator:

    def __init__(self, n_threads):
        # Making threads
        self.n_threads = n_threads

    def run_func(self, func, args_list):

        share = 1.0 * len(args_list) / self.n_threads

        threads = []

        # Separating arguments for each thread!
        for i in range(self.n_threads):
            threads.append(Worker(func, args_list[int(np.floor(i * share)):int(np.floor((i + 1) * share))]))

        # Running the threads
        for th in threads:
            th.start()

        # waiting for them to join
        res = []

        for th in threads:
            th.join()
            res += th.results

        return res


''' TEST
if __name__ == '__main__':
    wc = WorkersCoordinator(6)
    res = wc.run_func(lambda x, y: 10 * x + y, [(i, j) for i in range(5) for j in range(5)])
    res = [[res[i * 5 + j] for j in range(5)] for i in range(5)]
    print(res)
'''
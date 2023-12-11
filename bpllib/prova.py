import multiprocessing


def square(x):
    return x * x


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    results = pool.map(square, range(10))
    print(results)

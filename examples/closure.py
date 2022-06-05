from functools import partial


def work(b):
    def closure(a):
        print(b)

    return closure


if __name__ == '__main__':
    pp = [3]
    c1 = work(pp)
    pp.append(4)
    c2 = work(pp)
    print(c1(2), c2(2))

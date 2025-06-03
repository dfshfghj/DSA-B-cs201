def euler_sieve(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 10002):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > 10001:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes

def is_prime(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0:
        return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    if n < 2047:
        bases = [2]
    elif n < 1_373_653:
        bases = [2, 3]
    elif n < 25_326_001:
        bases = [2, 3, 5]
    elif n < 3_215_031_751:
        bases = [2, 3, 5, 7]
    else:
        bases = [2, 3, 5, 7, 11]

    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
if __name__ == '__main__':
    print(euler_sieve(10001))
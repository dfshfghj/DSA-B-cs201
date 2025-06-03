def bisect_left(x, lo, hi, check): # check: key(a[mid]) < x
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid, x):
            lo = mid + 1
        else:
            hi = mid
    return lo

def bisect_right(x, lo, hi, check): # check: x < key(a[mid])
    while lo < hi:
        mid = (lo + hi) // 2
        if check(x, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

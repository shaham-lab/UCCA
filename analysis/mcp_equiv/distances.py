import numpy as np
from bisect import bisect_left

def bubble_sort_distance(p):
    """Number of inversions in permutation p."""
    n = len(p)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inv_count += 1
    return inv_count

def get_all_distances(p):
    return {
        "bubble_sort": float(bubble_sort_distance(p))
    }

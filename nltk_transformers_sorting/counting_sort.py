# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:11:26 2024

@author: husse
"""


def counting_sort(arr):
    if not arr:
        return arr

    m, M = min(arr), max(arr)
    size = M - m + 1
    count_arr = [0] * size

    for n in arr:
        count_arr[n-m] += 1

    for i in range(1, size):
        count_arr[i] += count_arr[i-1]

    sorted_arr = [0] * len(arr)

    for n in reversed(arr):
        idx = n - m
        count_arr[idx] -= 1
        sorted_arr[count_arr[idx]] = n
    return sorted_arr


arr = list(map(int, input().rstrip().split()))
print(counting_sort(arr))

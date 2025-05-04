# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:04:03 2024

@author: husse
"""
from random import randint

def quick_sort(arr):

    if len(arr) <= 1:
        return arr

    l = len(arr)
    pivot = randint(0, l-1)
    L, R, sorted_arr = [], [], []

    for n in range(l):
        if n != pivot:

            if arr[n] <= arr[pivot]:
                L.append(arr[n])
            else:
                R.append(arr[n])
    L = quick_sort(L)
    R = quick_sort(R)

    sorted_arr.extend(L)
    sorted_arr.append(arr[pivot])
    sorted_arr.extend(R)

    return sorted_arr

arr = list(map(int, input().rstrip().split()))
print(quick_sort(arr))
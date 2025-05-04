# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:58:20 2024

@author: husse
"""


def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)
    return arr


def median_of_three(arr, low, high):
    mid = (low + high) // 2
    candidates = sorted([arr[low], arr[mid], arr[high]])
    median = candidates[1]
    if median == arr[low]:
        return low
    elif median == arr[mid]:
        return mid
    else:
        return high


def partition(arr, low, high):
    pivot_index = median_of_three(arr, low, high)
    pivot = arr[pivot_index]
    arr[low], arr[pivot_index] = pivot, arr[low]
    left, right = low + 1, high

    while True:
        while left <= right and arr[left] <= pivot:
            left += 1
        while left <= right and arr[right] > pivot:
            right -= 1
        if left > right:
            break
        arr[left], arr[right] = arr[right], arr[left]
    arr[low], arr[right] = arr[right], arr[low]
    return right


arr = list(map(int, input().rstrip().split()))
print(quicksort(arr))

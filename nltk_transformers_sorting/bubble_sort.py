def bubble_sort(arr):
    size = len(arr)
    swapped = False

    for i in range(size-1):
        for k in range(size-1-i):
            if arr[k] > arr[k+1]:
                arr[k], arr[k+1] = arr[k+1], arr[k]
                swapped = True
        if swapped is False:
            break
    print(arr)

arr = list(map(int, input().rstrip().split()))

bubble_sort(arr)

def selection_sort(arr):
    size = len(arr)

    for i in range(size):
        for k in range(i, size):
            if arr[k] == min(arr[i:size]):
                arr[i], arr[k] = arr[k], arr[i]
    print(arr)

arr = list(map(int, input().rstrip().split()))


selection_sort(arr)

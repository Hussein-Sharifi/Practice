def insertion_sort(arr):
    size = len(arr)
    for i in range(1, size):
        for k in range(i):

            if arr[i] > arr[k]:
                continue

            else:
                arr.insert(k, arr[i])
                del arr[i+1]
                break
    print(arr)

arr = list(map(int, input().rstrip().split()))
insertion_sort(arr)

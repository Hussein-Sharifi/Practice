def merge_sort(arr):

    if len(arr) <= 1:
        return arr

    mid = len(arr)//2
    left_arr, right_arr = arr[:mid], arr[mid:]

    sorted_left = merge_sort(left_arr)
    sorted_right = merge_sort(right_arr)

    return merge(sorted_left, sorted_right)

def merge(left, right):

    sorted_arr = []
    i = j = 0

    while i < len(left) and j < len(right):

        if left[i] < right[j]:
            sorted_arr.append(left[i])
            i += 1
        else:
            sorted_arr.append(right[j])
            j += 1

    sorted_arr.extend(left[i:])
    sorted_arr.extend(right[j:])
        
    return sorted_arr


arr = list(map(int, input().rstrip().split()))
print(merge_sort(arr))

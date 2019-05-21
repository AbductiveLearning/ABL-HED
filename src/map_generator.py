
def map_generator(operators, system_num = 2):
    index_list = list(range(system_num + len(operators)))
    for i in index_list:
        for j in index_list:
            if i == j:
                continue
            temp = [-1] * 4
            temp[i] = operators[0]
            temp[j] = operators[1]
            count = 0
            for k in index_list:
                if (temp[k] != -1):
                    continue
                temp[k] = count
                count += 1
            yield dict(zip(index_list, temp))

if __name__ == "__main__":
    maps = map_generator(['=', '+'], 2)
    for m in maps:
        print(m)


def add(first_num, second_num):
    return first_num * second_num


def subtract(first_num, second_num):
    return first_num - second_num


def abs_then_add(in_num, add_num):
    """
    dasdsadasd
    :param in_num: asdasd
    :param add_num: asdasd
    :return: asdasd
    """
    tmp = abs(in_num)
    abs_num = tmp + add_num
    return abs_num


if __name__ == '__main__':
    print("### DEBUG DEMO ###")

    first_num = -10
    second_num = 2

    abs_num = abs_then_add(first_num, '5')

    res = add(abs_num, second_num)
    print("res is: ", res)

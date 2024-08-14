
def string_to_list_of_type(data, type_, lower=False, sep=','):

    if lower:
        return list(type_(value.strip().lower()) for value in data.split(sep)) if sep != "" else list(
            type_(value.strip().lower()) for value in list(data))
    else:
        return list(type_(value.strip())
                    for value in data.split(sep)) if sep != "" else list(type_(value.strip()) for value in list(data))

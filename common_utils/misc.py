def tuplify(inputs):
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)

    return inputs

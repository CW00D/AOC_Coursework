def evaporate(test):
    test *= 10

test = {(1,1):1, (1,2):2, (1,3):3, (2,1):4, (2,2):5, (2,3):6}
evaporate(test)
print(test)
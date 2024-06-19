import numexpr as ne

def calculate(function):
    try:
        result = ne.evaluate(function).reshape(1,-1)[0][0]
        return str(function) + '的计算结果为' + str(result)
    except:
        return '很抱歉，我暂时无法进行此计算。'
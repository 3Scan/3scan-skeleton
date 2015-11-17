from functools import lru_cache

import numpy as np

from pyeda.inter import exprvar
from Thin3dtemplates import firstSubIter

"""
   To avoid pyeda overhead and evaluating the
   expressions each time,get a boolean
   expression in advance and evaluate it
"""


def emitCodeFromEqtn(eqtn, eqName):
    simplified = eqtn.simplify()
    uniqueIDtoSymbol = {symbol.uniqid: symbol for symbol in eqtn.inputs}

    # args = ", ".join(["uint8 %s" % symbol.name for symbol in eqtn.inputs])

    # Emit the c function header
    # print("uint8 %s(%s) {" % (eqName, args))
    statement = recursiveEmitter(simplified.to_ast(), uniqueIDtoSymbol)
    # print("\treturn %s;" % statement)
    # print("}")
    return statement


def recursiveEmitter(ast, symbolTable):
    # Input ast format is (operator, expr, expr, expr, expr. . .)
    op = ast[0]
    exprs = ast[1:]

    if op == "and":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " & ".join(subExprs) + ")"

    elif op == "or":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " | ".join(subExprs) + ")"

    elif op == "not":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " not ".join(subExprs) + ")"

    elif op == "xor":
        subExprs = [recursiveEmitter(exp, symbolTable) for exp in exprs]
        return "(" + " ^ ".join(subExprs) + ")"

    elif op == "lit":
        # Lit can have only one operator
        symbol = exprs[0]

        if symbol < 0:
            symbol *= -1
            return "(not %s)" % symbolTable[symbol].name
        else:
            return symbolTable[symbol].name
    raise RuntimeError("No way to resolve operation named '%s' with %i arguments" % (op, len(exprs)))


@lru_cache()
def getExpression():
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(exprvar, 'abcdefghijklmnopqrstuvwxyz')
    origin = exprvar('origin')
    validateMatrix = np.array([[[a, b, c], [d, e, f], [g, h, i]], [[j, k, l], [m, origin, n], [o, p, q]], [[r, s, t], [u, v, w], [x, y, z]]])
    usDeletiondirection, str1 = firstSubIter(validateMatrix)
    us = emitCodeFromEqtn(usDeletiondirection, usDeletiondirection)
    return us


us = getExpression()
print(us)

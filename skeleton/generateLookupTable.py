import numpy as np

from pyeda.inter import exprvar

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


def getExpression():
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = map(exprvar, 'abcdefghijklmnopqrstuvwxyz')
    usDeletiondirection = (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & p & (d | e | f | m | n | u | v | w | g | h | i | o | q | x | y | z)) | \
                          (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & v & (r | s | t | j | k | l | m | n | u | w | o | p | q | x | y | z)) | \
                          (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & y & (m | n | u | w | o | q | x | z)) | \
                          (~(a) & ~(b) & ~(c) & ~(k) & ~(e) & ~(d & j) & ~ (l & f) & p & v) | \
                          (~(a) & ~(b) & ~(k) & ~(e) & c & v & p & ~(j & d) & (l ^ f)) | \
                          (a & v & p & ~(b) & ~(c) & ~(k) & ~(e) & ~(l & f) & (j ^ d)) | \
                          (~(a) & ~(b) & ~(k) & ~(e) & n & v & p & ~(j & d)) | \
                          (~(b) & ~(c) & ~(k) & ~(e) & m & v & p & ~(l & f)) | \
                          (~(b) & ~(k) & ~(e) & a & n & v & p & (j ^ d)) | \
                          (~(b) & ~(k) & ~(e) & c & m & v & p & (l ^ f)) | \
                          (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(d) & ~(e) & ~(g) & ~(h) & q & y) | \
                          (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(l) & ~(r) & ~(s) & ~(t) & ~(e) & ~(f) & ~(h) & ~(i) & o & y) | \
                          (~(a) & ~(b) & ~(c) & ~(j) & ~(k) & ~(r) & ~(s) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & w & y) | \
                          (~(a) & ~(b) & ~(c) & ~(d) & ~(e) & ~(f) & ~(g) & ~(h) & ~(i) & ~(k) & ~(l) & ~(s) & ~(t) & u & y)
    us = emitCodeFromEqtn(usDeletiondirection, usDeletiondirection)
    return us


us = getExpression()
# print(us)


alphabet = "abcdefghijklmnopqrstuvwxyz"


def generateLookupTablearray(start, stop):
    lookupTablearray = np.zeros(start:stop, dtype=np.uint8)
    for i, value in enumerate(lookupTablearray):
        if i > 40000 :
            break
        # Don't re-generate values we've already calculated.
        neighborValues = [(i >> digit) & 0x01 for digit in range(26)]
        if sum(neighborValues) == 1:
            lookupTablearray[i] = 0
        else:
            neighborDict = dict(zip(alphabet, neighborValues))
            lookupTablearray[i] = eval(us, neighborDict)
    return lookupTablearray


def main():
    lookupTablearray = generateLookupTablearray()
    np.save("lookupTablearray.npy", lookupTablearray)


if __name__ == "__main__":
    main()

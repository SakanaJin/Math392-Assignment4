from copy import deepcopy
from math import sqrt

TOLERANCE = 1e-3

def matmul(A, B):
    if type(B[0]) == list:
        return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    return [sum(A[i][j]*B[j] for j in range(len(B))) for i in range(len(A))]

def apply_perm(Pvec: list[int], b: list[float]) -> list[float]:
    return [b[Pvec[i]] for i in range(len(b))]

def identity(n: int) -> list[list[float]]:
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def is_close(A: list[list[float]], B: list[list[float]], *, tolerance=1e-6) -> bool:
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > tolerance:
                return False
    return True

def transpose(A: list[list[float]]) -> list[list[float]]:
    alpha = deepcopy(A)
    n = len(alpha)
    for i in range(n):
        alpha[i] = [A[j][i] for j in range(n)]
    return alpha

def is_diagonally_dominant(A: list[list[float]]) -> bool:
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < sum([abs(A[i][j]) for j in range(n) if j != i]):
            return False
    return True

def is_SPD(A: list[list[float]]) -> bool:
    n = len(A)
    if not A == transpose(A):
        return False
    L = [[0 for _ in range(n)] for _ in range(n)]
    # cholesky check
    try:
        for i in range(n):
            L[i][i] = sqrt(A[i][i] - sum([L[i][k]**2 for k in range(i)]))
            for j in range(i+1, n):
                L[j][i] = (A[i][j] - sum([L[i][k] * L[j][k] for k in range(i)])) / L[i][i]
        if not is_close(A, matmul(L, transpose(L))):
            return False
        else: 
            return True
    except:
        return False
    
def spectral_radius(A: list[list[float]]) -> float:
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j <= i:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
    Linv = inverse(L)
    Linv = [[-Linv[i][j] for j in range(n)] for i in range(n)]
    Eigendiag = matmul(Linv, U)
    return max([abs(Eigendiag[i][i]) for i in range(n)])

def augment(A: list[list[float]], B: list[list[float]] | list[float]) -> list[list[float]]:
    if len(A) != len(B):
        raise ValueError("A and B must contain same number of rows")
    n = len(A)
    AM = deepcopy(A)
    if isinstance(B, list) and isinstance(B[0], list):
        AM = [a + b for a, b in zip(A, B)]
    else:
        for i in range(n):
            AM[i].append(B[i])
    return AM

def rowdiv(A: list[list[float]], row: int, val: float) -> None:
    A[row] = [A[row][i] / val for i in range(len(A[row]))]

def rowop(A: list[list[float]], row1: int, row2: int, val:float) -> None:
    '''R2 - val * R1 -> R2'''
    A[row2] = [A[row2][i] - (val * A[row1][i]) for i in range(len(A[row2]))]

def backwards_sub(b: list[float], U: list[list[float]]) -> list[float]:
    n = len(b)
    b[-1] = b[-1] / U[-1][-1]
    for i in range(n-2, -1, -1):
        for j in range(n-1, i, -1):
            U[i][j] = U[i][j] * b[j]
        b[i] = (b[i] - sum(U[i][i+1:n]))/U[i][i]
    return b

def forwards_sub(b: list[float], L: list[list[float]]) -> list[float]:
    n = len(b)
    for i in range(1, n):
        for j in range(0, i):
            L[i][j] = L[i][j] * b[j]
        b[i] = b[i] - sum(L[i][0:i])
    return b

def Gauss_elim(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(A)
    AM = augment(A, b)
    for i in range(n):
        for j in range(i+1, n):
            val = AM[j][i] / AM[i][i]
            rowop(AM, i, j, val)
    U = [row[:n] for row in AM]
    newb = [row[n] for row in AM]
    return backwards_sub(newb, U)

def inverse(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    AM = augment(A, identity(n))
    for i in range(n):
        for j in range(i+1, n):
            val = AM[j][i] / AM[i][i]
            rowop(AM, i, j, val)
    for i in range(n-1, 0, -1):
        for j in range(i-1, -1, -1):
            val = AM[j][i] / AM[i][i]
            rowop(AM, i, j, val)
        rowdiv(AM, i, AM[i][i])
    rowdiv(AM, 0, AM[0][0])
    Ainv = [row[n:] for row in AM]
    return Ainv

def inverse_solve(A:list[list[float]], b: list[float]) -> list[float]:
    Ainv = inverse(A)
    return matmul(Ainv, b)

def PvecLU(A: list[list[float]]) -> tuple[list[int], list[list[float]], list[list[float]]]: #this was something I had already made, it's OKAY 
    n: int = len(A)
    m: int = len(A[0])
    if any(m != len(row) for row in A):
        raise ValueError("All rows must have same number of columns")
    alpha = deepcopy(A)
    Pvec = list(range(n))
    row = 0
    for pivot in range(min(n, m)):
        row = pivot
        while row < n and abs(alpha[row][pivot]) < 1e-12:
            row += 1
        if row == n:
            raise ValueError("No valid pivot point in column {}".format(pivot))
        if row != pivot:
            alpha[pivot], alpha[row] = alpha[row], alpha[pivot]
            Pvec[pivot], Pvec[row] = Pvec[row], Pvec[pivot]
        for i in range(pivot+1, n):
            alpha[i][pivot] /= alpha[pivot][pivot]
            alpha[i][pivot+1:] = [alpha[i][j] - alpha[i][pivot] * alpha[pivot][j] for j in range(pivot+1, m)]
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(min(n, m)):
        for j in range(i, m):
            U[i][j] = alpha[i][j]
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i+1, n):
            L[j][i] = alpha[j][i]
    return (Pvec, L, U)

def PLUSolve(Pvec: list[int], L: list[list[float]], U: list[list[float]], b: list[float]) -> list[float]:
    b = apply_perm(Pvec, b)
    b = forwards_sub(b, L)
    b = backwards_sub(b, U)
    b = apply_perm(Pvec, b)
    return b

def will_Gauss_Seidell_Converge(A: list[list[float]]) -> bool:
    if is_diagonally_dominant(A):
        return True
    if is_SPD(A):
        return True
    if spectral_radius(A) < 1:
        return True
    return False

def Gauss_Seidell(A: list[list[float]], b: list[float], *, x0=None, max_iter=100) -> list[float]:
    n = len(A)
    if x0 is None:
        x = [0]*n
    else:
        x = x0[:]
    for _ in range(max_iter):
        oldx = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * oldx[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        if max([abs(x[i] - oldx[i]) for i in range(n)])/max([abs(x[i]) for i in range(n)]) < TOLERANCE:
            return x
    raise ValueError("Too many iterations :(")

if __name__ == "__main__":
    A: list[list[float]] = [[7, -3, 2, 5, -1, 4, -2, 6, 3, -4],
                            [2, 8, -1, 3, 5, -2, 4, -3, 1, 6],
                            [-4, 1, 9, -2, 3, 6, -5, 2, -1, 4],
                            [3, -2, 4, 7, -3, 1, 6, -4, 2, -5],
                            [1, 5, -3, 4, 8, -4, 2, 3, -2, 1],
                            [-2, 3, 6, -1, 4, 9, -3, 5, 4, -6],
                            [4, -1, 2, 6, -2, 3, 7, -5, 3, -2],
                            [5, 4, -4, 2, 6, -1, 3, 8, -4, 3],
                            [-3, 2, 1, -4, 5, 4, -6, 3, 9, -7],
                            [2, -6, 3, 5, -4, 2, 4, -1, 5, 10]]
    
    b: list[float] = [67, 113, 80, 8, 64, 87, 56, 112, 30, 176] #112 was originally 109

    xtrue = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # spd_test = [[4,12,-16], #should be true
    #         [12,37,-43],
    #         [-16,-43,98]]
    
    # spectral_test = [[1,2], [3,4]] #should be 1.5

    GaussSol = Gauss_elim(A, b)
    InverseSol = inverse_solve(A, b)
    Pvec, L, U = PvecLU(A)
    LUSol = PLUSolve(Pvec, L, U, b)
    GSSol = None
    if will_Gauss_Seidell_Converge(A):
        try:
            GSSol = Gauss_Seidell(A, b, max_iter=1000)
        except ValueError as e:
            print(e)
    else:
        print("Gauss-Seidell will not converge for matrix A")

    print("xtrue =", ['{:.2f}'.format(num) for num in xtrue])
    print("GaussSol =", ['{:.2f}'.format(num) for num in GaussSol])
    print("InverseSol =", ['{:.2f}'.format(num) for num in InverseSol])
    print("LUSol =", ['{:.2f}'.format(num) for num in LUSol])
    if GSSol:
        print("GSSol =", ['{:.2f}'.format(num) for num in GSSol])
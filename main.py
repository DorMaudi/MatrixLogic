from colors import bcolors
import matrix_utility as mtx
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""

def matrix_inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]: # 0 -> rows, 1 -> col.
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    if np.linalg.det(matrix) == 0: raise ValueError("cannot find its inverse.")
    #if mtx.Determinant(matrix, 1) == 0: raise ValueError("cannot find its inverse.")

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        elemn_matrix = np.identity(n)
        if matrix[i, i] == 0:
            j = i
            while j < n-1:
                if matrix[j+1, i] != 0:  # Check if there's a row below to swap with
                    mtx.swap_rows_elementary_matrix(elemn_matrix, i, j + 1)
                    matrix = np.dot(elemn_matrix, matrix)
                    identity = np.dot(elemn_matrix, identity)
                    print(bcolors.HEADER, "ID:\n", identity, bcolors.ENDC)
                    print(bcolors.WARNING, f"Matrix row swap {j} with {j + 1}\n", matrix, bcolors.ENDC)
                    j += 1
                    break
                elif matrix[j+1, i] == 0:
                    j += 1
                    continue
                else:
                    raise ValueError("Matrix is singular, cannot find its inverse.")



        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = mtx.scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)
            print(bcolors.HEADER, "ID:\n", identity, bcolors.ENDC)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j and matrix[j, i] != 0:
                scalar = -matrix[j, i]
                elementary_matrix = mtx.row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)
                print(bcolors.HEADER, "ID:\n", identity, bcolors.ENDC)

    return identity


if __name__ == '__main__':

    A = np.array([[4, 2, 3],
                  [2, 1, 4],
                  [1, 2, 3]])

    B = np.array([4, 2, 3])

    np.set_printoptions(suppress=True, precision=4)

    try:
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)
        #print(bcolors.OKGREEN, "Test\n", np.dot(A, A_inverse), "\n", bcolors.ENDC)
        print(bcolors.OKGREEN, "Solving for -> X, Y, Z...:\n", np.dot(A_inverse, B), bcolors.ENDC)

    except ValueError as e:
        print(str(e))

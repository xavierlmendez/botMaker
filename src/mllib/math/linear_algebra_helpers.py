import numpy as np


class QuadraticFormHelper:
    metadata = {
        "name": "Quadratic Form Helper",
        "description": "Helper for constructing quadratic form matrices.",
    }

    # TODO(BL-16): derive metadata by introspection
    def compute_q(self, vector_function):
        # Q = X.T A X where A is an n x n symmetric matrix that uniquely represents the quadratic form
        # this is used to simpligy multivariable functions via being a matrix representation for degree-two polynomials
        # for the ml use case I am taking advantage to aid in expanding the hypothesis space
        Q = np.diag(vector_function**2)
        return Q

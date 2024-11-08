import itertools
import operator
import functools
from numpy.typing import ArrayLike

def compute_cartesian_products(seasonalities: ArrayLike, N: int):
    """
    Compute the Cartesian product of distinct seasonalities and calculate their products.

    Args:
    - seasonalities (list): List of seasonalities.
    - N (int): Series length.

    Returns:
    - list: List of original seasonalities and their Cartesian products.
    """
    # Input validation
    if len(seasonalities): 
        try:
            # Attempt to convert seasonalities to a list
            seasonalities = list(seasonalities)
        except TypeError:
            raise ValueError("Input 'seasons' must be convertible to a list")

        # Check if all elements in the seasonalities list can be converted to integers
        try:
            seasonalities = [int(s) for s in seasonalities]
        except ValueError:
            raise ValueError("All elements in the 'seasons' list must be convertible to integers")

    # Generate all possible combinations of seasonalities
    combinations = []
    for r in range(1, len(seasonalities) + 1):
        combinations.extend(itertools.combinations(seasonalities, r))

    # Compute the product of each combination
    products = []
    for combination in combinations:
        product = functools.reduce(operator.mul, combination, 1)
        products.append(product)

    # Combine original seasonalities with their Cartesian products (using set to remove duplicates)
    result = list(set(seasonalities + products))
    result = [x for x in result if x <= N]

    return result
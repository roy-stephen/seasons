import itertools
import operator
import functools

def compute_cartesian_products(seasonalities):
    """
    Compute the Cartesian product of distinct seasonalities and calculate their products.

    Args:
    - seasonalities (list): List of seasonalities.

    Returns:
    - list: List of original seasonalities and their Cartesian products.
    """
    # Generate all possible combinations of seasonalities
    combinations = []
    for r in range(1, len(seasonalities) + 1):
        combinations.extend(itertools.combinations(seasonalities, r))

    # Compute the product of each combination
    products = []
    for combination in combinations:
        product = functools.reduce(operator.mul, combination, 1)
        products.append(product)

    # Combine original seasonalities with their Cartesian products
    result = seasonalities + [product for product in products if product not in seasonalities]

    return result

__all__ = [compute_cartesian_products]
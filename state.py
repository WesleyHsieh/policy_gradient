class State:

    """
    Wrapper class for state representation.
    """

    def __init__(self, grasp_coords, bounds, objs, target):
        self.grasp_coords = grasp_coords
        self.bounds = bounds
        self.objs = objs
        self.target = target

    def out_of_bounds(self):
        """
        Calculates number of objects knocked out of bounds.

        Output:
        n: int
            Number of objects out of bounds.
        """
        return sum([self.check_bounds(obj) for obj in self.objs])

    def check_bounds(self, obj):
        """
        Calculates number of objects knocked out of bounds.

        Parameters:
        obj: object
            Object to check.

        Output:
        n: int
            Number of objects out of bounds.
        """
        pass


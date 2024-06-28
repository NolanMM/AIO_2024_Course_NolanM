import numpy as np


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words


class Levenshtein:
    def __init__(self, s1, s2):
        """
        Initialize the Levenshtein object with two strings.

        Parameters:
        s1 (str): The first string.
        s2 (str): The second string.
        """
        self.s1 = str(s1)
        self.s2 = str(s2)
        self.rows = len(self.s1) + 1
        self.cols = len(self.s2) + 1

        # Define Result
        self.distance = self.checking_input()
        if self.distance == 0:
            self.distance = self.compute_distance()

    def checking_input(self):
        if self.rows == 1 and self.cols != 1:
            return self.cols - 1
        elif self.cols == 1 and self.rows != 1:
            return self.rows - 1
        else:
            return 0

    def compute_distance(self):
        """
        Compute the Levenshtein distance between the two strings.

        Returns:
        int: The Levenshtein distance.
        """

        # Initialize the distance matrix with zeros
        self.distance_matrix = np.zeros((self.rows, self.cols), dtype=int)

        # Initialize the first column and first row of the matrix
        for i in range(1, self.rows):
            self.distance_matrix[i][0] = i
        for j in range(1, self.cols):
            self.distance_matrix[0][j] = j

        # Populate the distance matrix
        for i in range(1, self.rows):
            for j in range(1, self.cols):
                cost = 0 if self.s1[i - 1] == self.s2[j - 1] else 1
                self.distance_matrix[i][j] = min(
                    # lev.(a,b)(i-1,j)             + 1        # Deletion
                    self.distance_matrix[i - 1][j] + 1,
                    # lev.(a,b)(i,j-1)             + 1        # Insertion
                    self.distance_matrix[i][j - 1] + 1,
                    # lev.(a,b)(i-1,j-1)               + cost(0 or 1 per cost)        # Substitution
                    self.distance_matrix[i - 1][j - 1] + cost)
        return self.distance_matrix[self.rows - 1][self.cols - 1]

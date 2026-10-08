import numpy as np

class NativeTerrainSampler:
    def __init__(self, terrain):
        self.triangles = np.array([[terrain.Points[key] for key in face] for face in terrain.Faces["Visible"]]) / 1000
        self.minimum = self.triangles[:, :, :2].min(axis=1)
        self.maximum = self.triangles[:, :, :2].max(axis=1)

    def elevation(self, easting, northing):
        query = np.array([easting, northing])
        indices = np.flatnonzero(np.all(query >= self.minimum - 1e-8, axis=1) & np.all(query <= self.maximum + 1e-8, axis=1))
        for index in indices:
            triangle = self.triangles[index]
            coefficients = np.linalg.solve((triangle[1:, :2] - triangle[0, :2]).T, query - triangle[0, :2])
            if np.min(coefficients) >= -1e-9 and np.sum(coefficients) <= 1 + 1e-9:
                return float(triangle[0, 2] + coefficients @ (triangle[1:, 2] - triangle[0, 2]))
        raise ValueError(f"No saved native terrain facet at {easting}, {northing}")



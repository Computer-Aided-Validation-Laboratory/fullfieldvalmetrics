import numpy as np

def construct_orthogonal_coordinate_system(points):
    """
    Constructs an orthogonal coordinate system centered at the point cloud's centroid,
    with the normal to the fitted plane being the z-axis.

    Args:
        points: A NumPy array of shape (n, 3) representing the point cloud.

    Returns:
        A tuple (origin, x_axis, y_axis, z_axis) representing the coordinate system, where:
            - origin: The centroid of the point cloud.
            - x_axis, y_axis, z_axis: Orthogonal unit vectors representing the axes.
    """

    if points.shape[0] < 3:
        raise ValueError("At least 3 points are needed to fit a plane.")

    # Calculate the centroid
    origin = np.mean(points, axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(points.T)

    # Perform SVD
    u, s, v = np.linalg.svd(covariance_matrix)

    # The normal vector (z-axis) is the eigenvector corresponding to the smallest eigenvalue
    z_axis = v[-1]

    # The other two axes are the other eigenvectors from the SVD.
    x_axis = v[0]
    y_axis = v[1]

    # Ensure that the coordinate system is right-handed.
    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
        y_axis = -y_axis

    return origin, x_axis, y_axis, z_axis

# Example usage:
if __name__ == "__main__":
    # Generate example point cloud data
    num_points: int = 1_000_000_000
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    z = 2 * x + 3 * y + 1 + 0.1 * np.random.randn(num_points)
    points = np.column_stack((x, y, z))

    # Construct the coordinate system
    origin, x_axis, y_axis, z_axis = construct_orthogonal_coordinate_system(points)

    print("Origin (centroid):", origin)
    print("X-axis:", x_axis)
    print("Y-axis:", y_axis)
    print("Z-axis (normal):", z_axis)

    #Verify Orthogonality
    print("x.y dot product: ", np.dot(x_axis,y_axis))
    print("x.z dot product: ", np.dot(x_axis,z_axis))
    print("y.z dot product: ", np.dot(y_axis,z_axis))

    #Verify right handedness
    print("Cross(x,y).z :", np.dot(np.cross(x_axis,y_axis), z_axis))
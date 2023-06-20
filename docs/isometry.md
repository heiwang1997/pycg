# pycg.isometry

## Classes

### `Isometry`

The `Isometry` class represents an isometry transformation, which consists of a rotation represented by a quaternion and a translation represented by a 3-dimensional vector. The class provides various methods for creating, manipulating, and operating on isometry transformations.

#### Constructor

- `Isometry(q=None, t=None)`: Creates a new `Isometry` object with the given rotation quaternion `q` (default is the identity quaternion) and translation vector `t` (default is the zero vector).

#### Properties

- `rotation`: Returns an `Isometry` object with only the rotation part of the transformation.
- `matrix`: Returns the transformation matrix corresponding to the isometry.
- `continuous_repr`: Returns the continuous representation of the isometry as a 9-dimensional vector.
- `full_repr`: Returns the full representation of the isometry as a 12-dimensional vector.

#### Methods

- `from_matrix(mat, t_component=None, ortho=False)`: Creates an `Isometry` object from a transformation matrix `mat`. If `t_component` is specified, it represents the translation component of the matrix. If `ortho` is set to `True`, the rotation matrix will be projected onto the orthogonal group.
- `from_axis_angle(axis, degrees=None, radians=None, t=None)`: Creates an `Isometry` object from an axis-angle representation. The rotation axis can be specified as a 3-dimensional vector or as a string ("X", "Y", "Z" or "+X", "+Y", "+Z", "-X", "-Y", "-Z").
- `from_euler_angles(a1, a2, a3, format='XYZ', t=None)`: Creates an `Isometry` object from Euler angles. The Euler angles can be specified in degrees or radians. The `format` parameter determines the rotation order.
- `from_twist(xi)`: Creates an `Isometry` object from a twist vector, which represents a spatial velocity in the Lie algebra of SE(3).
- `from_so3_exp(phi)`: Creates an `Isometry` object from a rotation vector in the Lie algebra of SO(3) using the exponential map.
- `inv()`: Returns the inverse of the isometry.
- `to_gl_camera()`: Returns a new `Isometry` object that is adjusted for OpenGL camera coordinates.
- `look_at(source, target, up=None)`: Creates an `Isometry` object that represents the camera transformation from a source point to a target point, with an optional up vector.
- `chordal_l2_mean(*isometries)`: Calculates the mean isometry from a list of isometries using the chordal L2 mean method.
- `copy(iso)`: Creates a copy of the given `Isometry` object.
- `adjoint_matrix()`: Returns the adjoint matrix of the isometry.
- `log()`: Calculates the logarithm map of the isometry, which represents the twist vector in the Lie algebra of SE(3).
- `tangent(prev_iso, next_iso)`: Calculates the tangent vector between two isometries `prev_iso` and `next_iso` at the current isometry.
- `interpolate(source, target, alpha)`: Interpolates between two isometries `source` and `target` using a parameter `alpha` in the range [0, 1].

### `ScaledIsometry`

The `ScaledIsometry` class represents a scaled isometry transformation, which is an isometry transformation with an additional scaling factor applied outside the transformation. The class provides similar functionality to the `Isometry` class, with support for scaling.

#### Constructor

- `ScaledIsometry(s=1.0, iso=None)`: Creates a new `ScaledIsometry` object with the given scaling factor `s` (default is 1.0) and the `Isometry` object `iso` (default is the identity isometry).

#### Properties

- `t`: Returns the translation vector of the underlying `Isometry` object.
- `q`: Returns the rotation quaternion of the underlying `Isometry` object.

#### Methods

- `from_matrix(mat, ortho=False)`: Creates a `ScaledIsometry` object from a transformation matrix `mat`. If `ortho` is set to `True`, the rotation matrix will be projected onto the orthogonal group.
- `inv()`: Returns the inverse of the scaled isometry.
- `matrix`: Returns the transformation matrix corresponding to the scaled isometry.
- `__matmul__(other)`: Overloaded matrix multiplication operator (`@`) for operating on other isometries or transformations.

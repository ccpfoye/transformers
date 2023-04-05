import tensorflow as tf

# Scalars, vectors, and matrices
scalar = tf.constant(42)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])

# Rank, shape, and dtype
print("Rank:", tf.rank(matrix))
print("Shape:", matrix.shape)
print("Dtype:", matrix.dtype)

# Creating tensors
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Manipulating tensors
# Element-wise addition
c = a + b
print("Element-wise addition:", c)

# Indexing and slicing
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accessing an individual element
element = matrix[1, 1]
print("Element at (1, 1):", element.numpy())

# Slicing a sub-tensor
sub_matrix = matrix[0:2, 1:3]
print("Sub-matrix:", sub_matrix.numpy())

# Reshaping tensors
tensor = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_tensor = tf.reshape(tensor, (2, 3))
print("Reshaped tensor:", reshaped_tensor.numpy())

# Broadcasting
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([10, 20, 30])

# Broadcasting addition
c = a + b
print("Broadcasted addition:", c.numpy())

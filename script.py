import tensorflow as tf

print('***** Print tensors *****')
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

print('***** Run tensors in session *****')
sess = tf.Session()
print(sess.run([node1, node2]))

print('***** Create tensor collection and run with operation "add" *****')
node3 = tf.add(node1, node2)
print(node3)
print(sess.run(node3))

print('***** Use placeholders to promise future values *****')
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

print('***** Add another operation *****')
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

print('***** Apply variables to an operation over an array *****')
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

print('***** Linear regression with loss model (error) *****')
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

print('***** Adjust model to perfectly fit the graph *****')
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
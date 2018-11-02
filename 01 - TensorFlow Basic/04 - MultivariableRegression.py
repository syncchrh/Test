import tensorflow as tf

x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]


x_matrixData = [[73, 80, 75],
                [93, 88, 93],
                [89, 91, 90],
                [96, 98, 100],
                [73, 66, 70]]

y_matrixData = [[152],
                [182],
                [180],
                [196],
                [142]]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


x_matirx = tf.placeholder(tf.float32, shape=[None, 3])
y_matrix = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

w_matrix = tf.Variable(tf.random_normal([3, 1], name='weight'))
b_matrix = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis_matrix = tf.matmul(x_matirx, w_matrix) + b_matrix

hypothesis = w1*x1 + w2*x2 + w3*x3 + b

cost = tf.reduce_mean(tf.square(hypothesis_matrix - y_matrixData))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(1000):

        cost_val, hy_val, _ = sess.run([cost, hypothesis_matrix, train_op], feed_dict={x_matirx:x_matrixData, y_matrix:y_matrixData})

        print(step, "Cost: ", cost_val, "Hypo: ", hy_val)

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("x1:80, x2:90, x3:100, Y:", sess.run(hypothesis_matrix, feed_dict={x_matirx:[[80, 90, 100]]}))




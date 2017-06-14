import tensorflow as tf
import numpy as np
from p2dist import p2dist


class P2Disttest(tf.test.TestCase):

    def test_forward(self):
        bb = 12
        mm = 30
        nn = 20
        dd = 60

        matA = np.random.randn(bb, mm, dd).astype(np.float32)
        matB = np.random.randn(bb, nn, dd).astype(np.float32)
        matB = matB.transpose(0, 2, 1)

        expected = np.zeros((bb, mm, nn))
        for b in xrange(bb):
            for m in range(mm):
                for n in range(nn):
                    aa = matA[b, m, :]
                    bb = matB[b, :, n]
                    dot = (aa - bb).dot(((aa - bb)))
                    expected[b, m, n] = dot

        tensorA = tf.Variable(matA, dtype=tf.float32)
        tensorB = tf.Variable(matB, dtype=tf.float32)

        actual = p2dist(tensorA, tensorB)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(actual)

        self.assertTrue((actual - expected).sum() < 1e-3)


if __name__ == '__main__':
    tf.test.main()

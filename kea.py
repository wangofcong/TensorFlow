import tensorflow as tf
from tensorflow import keras
from keras import datasets,layers,Sequential,metrics

(x,y),(x_test,y_test) = datasets.mnist.load_data()
print(x.shape,y.shape)
bachsize = 128
db = tf.data.Dataset.from_tensor_slices((x,y))
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
def pro(x,y):
    x = tf.cast(x,tf.float32)/255.
    y = tf.cast(y,tf.int32)
    return x,y
db = db.map(pro).shuffle(1000).batch(bachsize)
db_test = db_test.map(pro).batch(bachsize)
db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape,sample[1].shape)

model = Sequential([
layers.Dense(256,activation=tf.nn.relu),
layers.Dense(128,activation=tf.nn.relu),
layers.Dense(64,activation=tf.nn.relu),
layers.Dense(10)
])
opt = keras.optimizers.Adam(learning_rate=1e-3)
model.build(input_shape=[None,784])
def main():
    for epoch in range(20):
        # 测量数据封装
        acc_meter=metrics.Accuracy()
        loss_meter=metrics.Mean()
        for step in enumerate(db):
            y_onehot = tf.one_hot(step[1][1], depth=10)
            x = tf.reshape(step[1][0],[-1,784])
            with tf.GradientTape() as tade:
                logits = model(x)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
                loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))
                # 测量数据加载
                loss_meter.update_state(loss2)
            grads = tade.gradient(loss2,model.trainable_variables)

            opt.apply_gradients(zip(grads,model.trainable_variables))
            if step[0]%100==0:
                # 测量数据结果输出
                print(epoch,step[0],float(loss),loss_meter.result().numpy())
                # 更新测量数据
                loss_meter.reset_state()
            if step[0]%500==0:
                total = 0
                total_correct = 0
                acc_meter.reset_state()
                for step,(x,y) in enumerate(db_test):
                    x = tf.reshape(x,[-1,28*28])
                    logits = model(x)
                    prob = tf.nn.softmax(logits,axis=1)
                    pred = tf.argmax(prob,axis=1)
                    pred = tf.cast(pred,tf.int32)
                    correct = tf.equal(pred,y)
                    correct = tf.cast(correct,tf.int32)
                    correct = tf.reduce_sum(correct)
                    total_correct+=int(correct)
                    total+=x.shape[0]
                    acc_meter.update_state(y,pred)
                acc = total_correct/total
                print(step,'eva',acc,acc_meter.result().numpy())
if __name__ == '__main__':
    main()
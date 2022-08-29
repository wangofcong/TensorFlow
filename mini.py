import tensorflow as tf
from tensorflow import keras
from keras import datasets,layers,Sequential,metrics
# 加载数据
(x,y),(x_test,y_test) = datasets.mnist.load_data()
print(x.shape,y.shape)
# 设置每次运行的图片数
bachsize = 128
# 将图片数据转换为tensor
db = tf.data.Dataset.from_tensor_slices((x,y))
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
# 数据预处理
def pro(x,y):
    x = tf.cast(x,tf.float32)/255.
    y = tf.cast(y,tf.int32)
    return x,y
db = db.map(pro).shuffle(1000).batch(bachsize)
db_test = db_test.map(pro).batch(bachsize)
# 查看数据形状
db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape,sample[1].shape)
# 构建网络
model = Sequential([
layers.Dense(256,activation=tf.nn.relu),
layers.Dense(128,activation=tf.nn.relu),
layers.Dense(64,activation=tf.nn.relu),
layers.Dense(10)
])
# 构建优化器
opt = keras.optimizers.Adam(learning_rate=1e-3)
# 网络初始化
model.build(input_shape=[None,784])
# 训练网络
def main():
    for epoch in range(20):
        for step in enumerate(db):
            # 目标数据one-hot
            y_onehot = tf.one_hot(step[1][1], depth=10)
            # 元数据塑性 【b，28,28】=》【b，784】 网络输入
            x = tf.reshape(step[1][0],[-1,784])
            # 梯度下降，后向运算
            with tf.GradientTape() as tade:
                logits = model(x)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
                loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))
            grads = tade.gradient(loss2,model.trainable_variables)
            opt.apply_gradients(zip(grads,model.trainable_variables))
            # 打印输出
            if step[0]%100==0:
                print(epoch,step[0],float(loss),float(loss2))
        # 测试网络性能
        total = 0
        total_correct = 0
        for x,y in db_test:
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
        acc = total_correct/total
        print("acc",acc)
if __name__ == '__main__':
    main()
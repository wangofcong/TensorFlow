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
    x = tf.reshape(x,[-1,28*28])
    x = tf.squeeze(x)
    y = tf.cast(y,tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y
db = db.map(pro).shuffle(1000).batch(bachsize)
db_test = db_test.map(pro).batch(bachsize)
db_iter = iter(db)
sample = next(db_iter)
sample = next(db_iter)
sample = next(db_iter)
sample = next(db_iter)
sample = next(db_iter)
sample = next(db_iter)
sample = next(db_iter)
print(sample[0].shape,sample[1].shape)
print(sample[0])

model = Sequential([
layers.Dense(256,activation=tf.nn.relu),
layers.Dense(128,activation=tf.nn.relu),
layers.Dense(64,activation=tf.nn.relu),
layers.Dense(10)
])
model.build(input_shape=[None,784])
# 模型参数初始化
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=tf.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
# 模型训练
model.fit(db,epochs=20,validation_data=db_test,validation_steps=1)
# 模型测试
model.evaluate(db_test)
# 保存模型权重
model.save_weights('./checkpaoint/myw')
del model
model = Sequential([
layers.Dense(256,activation=tf.nn.relu),
layers.Dense(128,activation=tf.nn.relu),
layers.Dense(64,activation=tf.nn.relu),
layers.Dense(10)
])
model.build(input_shape=[None,784])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=tf.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
# 加载模型
model.load_weights('./checkpaoint/myw')
acc = model.evaluate(db_test)
model.save("./cc/yy")
import numpy as np
import matplotlib.pyplot as plt

# 矩阵设计
# 状态矩阵
A = np.array([[1, 1], [0, 1]])
# 状态误差矩阵
Q = np.array([[0.1, 0], [0, 0.1]])
# 测量矩阵
H = np.array([[1, 0], [0, 1]])
# 测量误差矩阵
R = np.array([[1, 0], [0, 1]])
# 单位矩阵
I = np.array([[1, 0], [0, 1]])


# 状态量设计
# x1为实际位置，x2为实际速度
def kalman(x1First, x2First, Z, Pposterior, Q, R):
    # 求解上一次后验结果值
    XposteriorLast = np.array([[x1First], [x2First]])
    # 求解上一次后验误差协方差矩阵
    PposteriorLast = Pposterior
    prior = []
    posterior = []
    posterior.append(XposteriorLast)
    for i in range(len(Z)):
        # 预测环节
        # 求解这一次先验估计值
        XpriorCur = np.dot(A, XposteriorLast)
        # 记录这一次先验估计值
        prior.append(XpriorCur)
        # 求解先验先验估计结果的协方差矩阵
        PpriorCur = np.dot(A, PposteriorLast)
        PpriorCur = np.dot(PpriorCur, A.T) + Q

        # 校正环节
        # 求解卡尔曼增益
        Kk = G(PpriorCur, R)
        # 求解这一次后验结果值
        XposteriorCur = backJustify(XpriorCur, Kk, Z[i])
        # 记录这一次后验结果值
        posterior.append(XposteriorCur)
        # 更新这一次误差协方差矩阵
        P = (I - np.dot(Kk, H))
        P = np.dot(P, PpriorCur)

        # 更新环节
        # 更新后验误差协方差矩阵
        PposteriorLast = P
        # 更新后验结果
        XposteriorLast = XposteriorCur
    return prior, posterior


def G(PpriorCur, R):
    tmp1 = np.dot(PpriorCur, H.T)
    tmp2 = np.dot(H, PpriorCur)
    tmp3 = np.dot(tmp2, H.T)
    tmp4 = tmp3 + R
    tmp4 = np.linalg.inv(tmp4)
    Kk = np.dot(tmp1, tmp4)
    return Kk


def backJustify(Xfristk, Kk, Zk):
    tmp1 = Zk - np.dot(H, Xfristk)
    tmp2 = np.dot(Kk, tmp1)
    Xlastk = Xfristk + tmp2
    return Xlastk


def initiation(x, y):
    np.random.seed(1)
    # Z = np.random.uniform(-0.2, 0.2, size=(x, y))
    return Z


def getActual(noise, x1, x2):
    # 获取真实状态的取样数
    m = noise.shape[0]
    # 建立第一次真实状态，x1为位置，x2为速度
    X = np.array([[x1], [x2]])
    # 建立真实状态值集合
    actual = []
    actual.append(X)
    for i in range(m):
        # 这里的[i]指的是让取出的行数不丢失维度
        # n = noise[[i], :].T
        X = np.dot(A, X) + noise[[i], :].T
        actual.append(X)
    return actual


def getMeasurement(actual, noise):
    meas = []
    m = noise.shape[0]
    for i in range(m):
        mea = np.dot(H, actual[i + 1]) + noise[[i], :].T
        meas.append(mea)
    return meas


def getLocation(tmp):
    locations = []
    for i in range(len(tmp)):
        location = np.squeeze(tmp[i][0])
        locations.append(location)
    return locations


# test
# Z = []
# Z.append(np.array([[2.866], [1.828]]))
# Z.append(np.array([[2.773], [1.656]]))
# # Z = np.array([[2.866, 1.820], [2.773, 1.656]])
# Pposterior = np.array([[1, 0], [0, 1]])
# prior, posterior = kalman(0, 1, Z, Pposterior, Q, R)
# print(prior)
# print(posterior)

np.random.seed(2)
t = np.array([[0.731 - 0.060, 1.332 + 0.204]]).T
outcome = np.dot(A, t)

# 获取实际状态噪声
m = 30
actualNoise = np.random.uniform(-0.2, 0.5, size=(m, 2))
# ae = np.array([[1], [2], [3], [4], [5]])
averageA = (1 / m) * np.dot(np.ones(shape=(1, m)), actualNoise)
actualNoise = actualNoise - averageA
varianceA = (1 / m) * np.squeeze(np.dot(actualNoise.T, actualNoise))

# 获取测量状态噪声
n = 30
measureNoise = np.random.uniform(-0.7, 0.7, size=(n, 2))
averageM = (1 / n) * np.dot(np.ones(shape=(1, n)), measureNoise)
measureNoise = measureNoise - averageM
varianceM = (1 / n) * np.squeeze(np.dot(measureNoise.T, measureNoise))

actual = getActual(actualNoise, 0, 1)
measurement = getMeasurement(actual, measureNoise)
# print(actual)
# print('----------------------------------------')
# print(measurement)
# print(varianceA)
# print(varianceM)

# 代入卡尔曼计算
Pposterior = np.array([[1, 0], [0, 1]])
prior, posterior = kalman(0, 1, measurement, Pposterior, varianceA, varianceM)
# print(prior)
# print(posterior)

xkc = np.dot(A, np.array([[0.731], [1.332]])) + np.array([[-0.060], [0.204]])
# print(xkc)

actual_locations = getLocation(actual)
measurement_locations = getLocation(measurement)
prior_locations = getLocation(prior)
posterior_locations = getLocation(posterior)
print(actual_locations)
print(measurement_locations)
print(prior_locations)
print(posterior_locations)

x1 = np.arange(0, 31)
x2 = np.arange(1, 31)
ym1 = actual_locations
ym2 = measurement_locations
ym3 = prior_locations
ym4 = posterior_locations

lines = plt.plot(x1, ym1, x2, ym2, x2, ym3, x1, ym4)
# lines = plt.plot(x, ym1)
# 设置线的属性
plt.setp(lines[0], linewidth=2)
plt.setp(lines[1], linewidth=2)
plt.setp(lines[2], linewidth=2)
plt.setp(lines[3], linewidth=2)
# 线的标签
plt.legend(('actual', 'measurement', 'prior', 'posterior'), loc='upper right')
# plt.legend(('actual', '2'), loc='upper right')
plt.title("comparison on location")
plt.xlabel("step")
plt.ylabel("location")
plt.show()

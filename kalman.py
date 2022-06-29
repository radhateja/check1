import numpy as np
import matplotlib.pylab as plt

iX = 0
iV = 1
NUMVARS = iV + 1


class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float,
                       accel_variance: float) -> None:
        
        # mean of state GRV(Gaussian random variable)
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iV] = initial_v
        
        self._accel_variance = accel_variance

        # covariance of state GRV(Gaussian random variable)
        self._P = np.eye(NUMVARS)

    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a

        F = np.eye(NUMVARS)         # creating identical matrix and size is based on the NUMVARS
        #print("F::",F) 

        F[iX, iV] = dt              # storing Delta-t value in F
        #print("F1::",F)

        new_x = F.dot(self._x)      # DOT product of F and x and storing in new_x
        #print("new_x::",new_x) 

        G = np.zeros((2, 1))        # Created 2x1 matrix and storing in G
        G[iX] = 0.5 * dt**2         # 0.005 is stored in G[0] position
        G[iV] = dt                  # 0.1 is stored in G[1] position
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance
        print("new_P::",new_P)
        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]





##>>>main funtion<<<##


real_x = 0.0
meas_variance = 0.1 ** 2
real_v = 0.5

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)          

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_vs = []

for step in range(NUM_STEPS):
    if step > 500:
        real_v *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v

    kf.predict(dt=DT)

    meas_variance = meas_variance
    meas_value = real_x + np.random.randn() * np.sqrt(meas_variance)
    
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value=real_x + np.random.randn() * np.sqrt(meas_variance), meas_variance=meas_variance)

    real_xs.append(real_x)
    real_vs.append(real_v)

print("meas_value::",meas_value)
print("meas_variance::",meas_variance)
print("real_x::",real_x)
print("real_v::",real_v)
print("Dt::",DT)
print("ranb_num::",-meas_value+real_x)
#print("real_xs::",real_xs)
#print("real_vs::",real_vs)

plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')

plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot(real_vs, 'b')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')

plt.show()
#plt.ginput(1)

import time
import torch as tor
from torch.autograd import Variable



xs = []

for i in range(5) :
    xs.append(tor.randn(2 ** i, 1000))


for x in xs :
    s = time.time()
    x.cuda()
    e = time.time()
    print (i, round(e-s, 5))
    print (type(x))


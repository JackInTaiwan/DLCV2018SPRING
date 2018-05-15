import time
import torch as tor
from ACGAN.model_3 import GN, DN
from torch.autograd import Variable


gn = GN()
gn = gn.cuda()

for i in range(8) :
    print (i)
    x = tor.randn(8, 512)
    x = Variable(x)
    x = x.cuda()
    s = time.time()
    p = gn(x)
    e = time.time()
    print (e-s)




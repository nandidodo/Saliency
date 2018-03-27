# so the aim here is to figure out if my data is log normally distribted
# first thing to do is draw points and see if it is actually represented
# as a quadratic downward sloping line on a log log plot

import numpy as np
import matplotlib.pyplot as plt

samples = np.random.lognormal(0,1,100000)
n,bins,patches = plt.hist(samples)
plt.show()
plt.loglog(n, bins[1:])
plt.show()

# oay great!?
# so how does this work. waht about a standard normal

samples = np.random.normal(0,1,100000)
n,bins,patches = plt.hist(samples)
plt.show()
plt.loglog(n, bins[1:])
plt.show()


# crap. in fact it is almost certainly likely to be a standard normal distribution!
# as defined bythe clt?
# so that's a great and pretty rubbish thing
# which implies that it shuold be modelled as a diffusive random walk
# if thatdata is even good at all, which it is not.
# I'm fairly not sure what to do abotu that to be hoenst
# dagnabbit! that's going tocause me some seriosuly annoying issues then
# if it is just standardly normally distributed
# even if I have to do some tests to confirm that
# it means that most likely the data is fairly useless. so dagnabbit
# atleast the result is boring. Althoguh it has some fairly statistical heft
# it is still boring, and that is irritating
# and I haven't actually done any reasonable analysis of the scanpath data itself
# but I suppose it could trivially be modelled as a randomwalk?
# but that would be boring statistically
# but perhaps still worth publishing?
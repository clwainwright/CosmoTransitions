from helper_functions import Nbspl
t = [-1,-1,-1,-1, -.5, 0, .5, 1, 1, 1, 1]
x = np.linspace(-1,1,500)
y = Nbspl(t,x, k=3)
plt.plot(x, y)
plt.xlabel(r"$x$")
plt.ylabel(r"$y_i(x)$")
plt.show()
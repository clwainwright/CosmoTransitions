from tunneling1D import SingleFieldInstanton
import matplotlib.pyplot as plt

# Thin-walled
def V1(phi): return 0.25*phi**4 - 0.49*phi**3 + 0.235 * phi**2
def dV1(phi): return phi*(phi-.47)*(phi-1)
profile = SingleFieldInstanton(1.0, 0.0, V1, dV1).findProfile()
plt.plot(profile.R, profile.Phi)

# Thick-walled
def V2(phi): return 0.25*phi**4 - 0.4*phi**3 + 0.1 * phi**2
def dV2(phi): return phi*(phi-.2)*(phi-1)
profile = SingleFieldInstanton(1.0, 0.0, V2, dV2).findProfile()
plt.plot(profile.R, profile.Phi)

plt.xlabel(r"Radius $r$")
plt.ylabel(r"Field $\phi$")
plt.show()
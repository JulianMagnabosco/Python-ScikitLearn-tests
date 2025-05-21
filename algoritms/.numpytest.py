import numpy as np

data = np.array([
  [1,2,3],
  [4,5,6],
  [7,8,9]])
data3d = np.array([[
  [1,2,3],
  [4,5,6],
  [7,8,9]],[
  [1,2,3],
  [4,5,6],
  [7,8,9]]])

print("\n----Tabla----")
print(data)
print(data.shape)

print("\n----Tabla3d----")
print(data3d)
print(data3d.shape)

print("\n----fila----")
print(data[0])
print(data[0].shape)

print("\n----columna----")
print(data[:,0])
print(data[:,0].shape)

print("\n----columna como Tabla----")
print(data[:,np.newaxis,0])
print(data[:,np.newaxis,0].shape)

print("\n----Tabla como tridimencional----")
print("\n(opcion1)")
print(data[np.newaxis])
print(data[np.newaxis].shape)
print("\n(opcion2)")
print(data[:,np.newaxis])
print(data[:,np.newaxis].shape)
print("\n(opcion3)")
print(data[:,:,np.newaxis])
print(data[:,:,np.newaxis].shape)

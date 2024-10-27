import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
import time
import math
from noise import pnoise2
from enum import Enum

from numerical import difference, operator

class AXIS:
    X = 1
    Y = 2
    Z = 3

class Fluid:
     def __init__(self, shape, *quantities, pressure_order=1, advect_order=3, delta_x, epsilon, delta_t, z1, externalForce, phi_rel, gamma_heat, gamma_vapor):
        #グリッドとソルバーの作成
        self.shape = shape
        self.time = 0.0
        self.epsilon = epsilon
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.M_air = 28.96 #g/mol
        self.M_W = 18.02 #g/mol
        self.gamma = 0.0065 #K/m
        self.one_atm = np.zeros(shape)
        self.one_atm[:,:,:] = 101300
        #１気圧＝101325パスカル=
        self.E = 1
        self.generalGasConstant = 8.31
        #J/molK
        self.T_ISA = 273
        self.gravity = 9.81
        #m/s2
        self.phi_rel = phi_rel
        self.gamma_heat = gamma_heat
        self.gamma_vapor = gamma_vapor
        self.alpha_A = 1.0e-3
        self.alpha_K = 1.0
        self.alpha_E = 1.0e-1
        self.vapor_map = np.zeros(shape[:2])
        self.heat_map = np.zeros(shape[:2])
        scale = 10.0
        for i in range(shape[0]):
             for j in range(shape[1]):
                  self.vapor_map[i,j] = pnoise2(i / scale, j / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0)
                  self.heat_map[i,j] = pnoise2(i / scale, j / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0)

        self.theta = np.zeros(shape)
        self.pressure = self.pressure_altitude_field()
        self.pressure_0 = np.zeros(shape)
        self.temperature_0 = np.zeros(shape)
        self.Rd = 287 #J/(kg K)
        self.latenthead = 2.5
        #J/kg
        def f(z, z1):
             if z*delta_x <= z1:
                  return self.T_ISA + self.gamma * z * self.delta_x
             else:
                  return self.T_ISA + (2 * self.gamma * z1 - self.gamma * z * self.delta_x) 
        def Temperature_altitude(z1):
            z = np.arrange(shape[2])
            z = z[None, None, :]
            for k in range(z1):
                 z[:, :, k] = f(k, z1)
            return z
        
        self.T_air = Temperature_altitude(z1)
        self.temperature = self.T_air
        #ここ高度とz1の兼ね合いがあいまいなので調整///
        self.dimensions = len(shape)
        print('shape and dimension calculated')
        # Prototyping is simplified by dynamically 
        # creating advected quantities as needed.
        self.quantities_clouddrop = quantities[0]
        self.quantities_raindrop = quantities[1]
        self.quantities_vapor = quantities[2]
        self.quantities_sum_water = quantities[0] + quantities[1] + quantities[2]
        self.quantities_dryair = 1.293
        #kg/m3
        self.externalForce = externalForce
        height_grid = np.arange(shape[2]).reshape(shape[2], 1, 1)
        # それを10x10の平面全体にブロードキャストして、高さの配列を生成
        self.height_grid = np.broadcast_to(height_grid, shape) * delta_x
        #高さはkm表示。そのためにdelta_xで調整。必要であれば追加
        initial_vapor_distribution = np.zeros(shape[2])

        for k in range(shape[2]):
             initial_vapor_distribution[k] = math.exp(-2.29e-4 * self.height_grid + 1)
        initial_vapor_distribution = np.broadcast_to(initial_vapor_distribution.reshape(shape[2], 1, 1), shape)
        self.quantities_vapor = initial_vapor_distribution

        for q in quantities:
            setattr(self, q, np.zeros(shape))
        print('setattr calculated')
        self.indices = np.indices(shape)
        #indices...行番号と列番号を格納した同じ形の行列を返す
        self.velocity = np.zeros((self.dimensions, *shape))
        #shapeと同じ形の行列を次元の個数(ベクトルの要素の数)用意する
        self.vorticity = np.zeros((self.dimensions, *shape))
        self.vorticity_confinement((self.dimensions, *shape))
        self.buoyancy = np.zeros((self.dimensions, *shape))

        laplacian = operator(shape, difference(2, pressure_order))
        #factorized...引数の行列AをLU分解して、Ax=bの解xを求める際に使用する.その際、bは後から与える
        print('operator calculated')
        start_time = time.time()
        #時間がかかっていたため計測した


        self.pressure_solver = factorized(laplacian)
        elapsed_time = time.time() - start_time
        x = shape[0]
        file_name = 'time_memo.txt'
        with open(file_name, "a") as file:
            file.write(f"x = {x}, elapsed_time = {elapsed_time:.4f} seconds\n")
        #計測した時間をメモ
        
        print('LU decomposed')
        
        self.advect_order = advect_order
        print('initialized')
     
     def get_value(self, x, y, z, q):
          if (x < 0):
              x = 0
          if (x > self.shape[0] -1):
              x = self.shape[0] - 1
          if (y < 0):
              y = 0
          if (y > self.shape[1] -1):
              y = self.shape[1] - 1
          if (z < 0):
              z = 0
          if (z > self.shape[2] -1):
              z = self.shape[2] - 1
          return q[x,y,z]

     def get_offset(axis):
          offsets = {
        AXIS.X: (0, 0.5, 0.5),
        AXIS.Y: (0.5, 0, 0.5),
        AXIS.Z: (0.5, 0.5, 0)
          }
          return offsets.get(axis, (0, 0, 0))  # デフォルト値Vec3f(0, 0, 0)     

     def get_interpolation(self, p, q):
         return self.trilinear_interp(p,q)
     
     def get_face_value(self, q, p, axis):
          offset = self.get_offset(axis)
          return self.get_interpolation(p-offset,q)
     
     def get_center_value(self,q,p):
         offset = (0.5,0.5,0.5)
         return self.get_interpolation(p-offset,q)
     
     def trilinear_interp(self, p, q):
         x = p[0]
         y = p[1]
         z = p[2]

         x0 = int(x)
         y0 = int(y)
         z0 = int(z)

         x1 = x0 + 1
         y1 = y0 + 1
         z1 = z0 + 1
         
         xd = x - x0
         yd = y - y0
         zd = z - z0
         c000 = self.get_value(x0,y0,z0,q)
         c001 = self.get_value(x0,y0,z1,q)
         c010 = self.get_value(x0,y1,z0,q)
         c011 = self.get_value(x0,y1,z1,q)
         c100 = self.get_value(x1,y0,z0,q)
         c101 = self.get_value(x1,y0,z1,q)
         c110 = self.get_value(x1,y1,z0,q)
         c111 = self.get_value(x1,y1,z1,q)

         c00 = c000 * (1.0 - xd) + c100 * xd
         c01 = c001 * (1.0 - xd) + c101 * xd
         c10 = c010 * (1.0 - xd) + c110 * xd
         c11 = c011 * (1.0 - xd) + c111 * xd
         c0 = c00 * (1.0 - yd) + c10 * yd
         c1 = c01 * (1.0 - yd) + c11 * yd
         c = c0 * (1.0 - zd) + c1 * zd
         return c

     def trace(self, orig, u, dt):
         dir = u * dt
         target = orig - dir
         return target

     def lin_solve(self, b, x, x0, a, c):                                            
          for i in range(1,self.shape[0]-1):
               for j in range(1,self.shape[1]-1):
                    for k in range(1,self.shape[2]-1):
                         x[i,j,k] = x0[i,j,k] + a * (x[i-1,j,k] + x[i+1,j,k] + x[i,j+1,k] + x[i,j-1,k] + x[i,j,k+1] + x[i,j,k-1]) / c
          return x

     def diffuse(self, b, x, x0, diff, dt):
         MAX = max(max(self.shape[0], self.shape[1]), self.shape[2])
         a = dt*diff*MAX**3
         return self.lin_solve(b, x, x0, x0, a, 1+6*a)
     
     def advect(self, b, d, d0, u, v, w, dt):
         dtx=dty=dtz=dt * max((self.shape[0], self.shape[1]), self.shape[2])
         v1 = np.copy(self.velocity)
         for i in range(self.shape[0]):
             for j in range(self.shape[1]):
                 for k in range(self.shape[2]):
                     if (i+1<self.shape[0]):
                         orig = (i+1.0, 0.5+j, 0.5+k)


     def rotate(self,vector_field):
            jacobian_shape_rotate = (self.dimensions, ) * 2
            partials_rotate = tuple(np.gradient(d) for d in getattr(self, vector_field))
            jacobian_rotate = np.stack(partials_rotate).reshape(*jacobian_shape_rotate, *self.shape)
            curl_mask = np.triu(np.ones(jacobian_shape_rotate, dtype=bool), k=1)
            rotation = (jacobian_rotate[curl_mask] - jacobian_rotate[curl_mask.T]).squeeze()
            print('rotation calculated')
            return rotation
        
     def divergence(self, vector_field):
            jacobian_shape_div = (self.dimensions, ) * 2
            jacobian_div = np.stack(partials_div).reshape(*jacobian_shape_div, *self.shape)
            partials_div = tuple(np.gradient(d) for d in getattr(self, vector_field))
            print('divergence calculated')
            return jacobian_div.trace()

     def grad(self, scalar_field):
            print('gradient calculated')
            return np.gradient(getattr(self, scalar_field))

        #getattrで属性を持ってきてから
     def vorticity_confinement(self):
            w = self.rotate(self, 'velocity')
            k = np.gradient(np.linalg.norm(w))
            norm_k = np.linalg.norm(k, axis=0)
            norm_k[norm_k == 0] = 1
            N = k / norm_k
            transpose_N = np.transpose(N, (1,2,3,0))
            transpose_w = np.transpose(w, (1,2,3,0))
            vorticity_confinement = self.epsilon * self.delta_x * (np.transpose(np.cross(transpose_N, transpose_w), (3,0,1,2)))
            self.velocity += self.delta_t * vorticity_confinement
            print('vorticity confinment calculated')
    
     def pressure_altitude_field(self):
        print('pressure altitude calculated')
        return ((self.T_ISA * np.ones_like(self.height_grid) - self.gamma * self.height_grid) ** 5.2561)
     #height_gridはその地点でのm

     def getMoleFraction(self, quantity):
        self.quantities_sum_water = self.quantities_clouddrop + self.quantities_raindrop + self.quantities_raindrop
        q = getattr(self, quantity)
        X_i = q / (self.quantities_dryair + self.quantities_sum_water)
        print('mole fraction calculated')
        return X_i

     def getAverageMolarMass(self):
        X_V = self.getMoleFraction(self, 'quantities_vapor')
        M_th = X_V * self.M_W + (1 - X_V) * self.M_air
        print('average molar mass calculated')
        return M_th

     def getIsentropicExponent(self):
        M_th = self.getAverageMolarMass(self)
        X_V = self.getMoleFraction(self, 'quantities_vapor')
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        print('isentropic exponent calculated')
        return gammma_th
    
     def compute_buoyancy(self):
        M_th = self.getAverageMolarMass(self)
        X_V = self.getMoleFraction(self, 'quantities_vapor')
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        T_th = np.zeros(self.shape)
        B = np.zeros((self.dimensions,*self.shape))
        #M_air...スカラー値
        #M_th...スカラー場
        #T_th...スカラー場
        #T_air...スカラー場
        #配列同士の積は求められないので、要素ごとに計算
        for i in range(gammma_th.shape[0]):
             for j in range(gammma_th.shape[1]):
                  for k in range(gammma_th.shape[2]):
                       T_th[i,j,k] = self.pressure_altitude_field()[i,j,k] ** gammma_th[i,j,k]
                       B[2,i,j,k] = self.gravity * (self.M_air * T_th[i,j,k] / (M_th[i,j,k] * self.T_air[i,j,k]))
        print('buoyancy calculated')
        return B
    
     def apply_b_external(self):
         b = self.compute_buoyancy
         e = self.externalForce
         self.velocity += self.delta_t * (b + e)
         print('buoyancy and external force applied')

     def getHeatCapacity(self):
         gamma_th = self.getAverageMolarMass(self)
         M_th = self.getAverageMolarMass(self)
         cp_th = np.zeros(self.shape)
         for i in range(cp_th.shape[0]):
             for j in range(cp_th.shape[1]):
                  for k in range(cp_th.shape[2]):
                       cp_th[i,j,k] = gamma_th[i,j,k] * self.generalGasConstant / (M_th[i,j,k] * (gamma_th[i,j,k] - 1))
         print('heatcapacity calculated')
         return cp_th
    
     def potential_temperature(self):
         p0 = self.pressure_0
         p1 = self.pressure
         kappa = self.Rd / self.getHeatCpacity(self)
         T0 = self.temperature_0
         theta = np.zeros(self.shape)
         for i in range(theta.shape[0]):
              for j in range(theta.shape[1]):
                   for k in range(theta.shape[2]):
                        theta[i,j,k] = ((p1[i,j,k]/p0[i,j,k]) ** kappa[i,j,k]) * T0
         print('potential temperature calculated')
         return theta

     def getSaturationRatio(self, pressure):
         p = getattr(self, pressure)
         T = self.temperature
         qvs = np.zeros(self.shape)
         for i in range(qvs.shape[0]):
            for j in range(qvs.shape[0]):
                for k in range(qvs.shape[0]):
                    qvs[i,j,k] = 380.16 / p[i,j,k] * math.exp(17.67 * T[i,j,k] / (T[i,j,k] + 243.50))
         print('saturation ratio calculated')
         return qvs
    
     def boundary_condition(self):
        m = 2
        self.quantities_vapor[:,:,0] = self.phi_rel * self.getSaturationRatio(self, 'one_atm')[:,:,0] * (self.gamma_vapor * (m * self.vapor_map - 1) + 1)
        self.temperature[:,:,0] = self.T_ISA + self.E * (self.gamma_heat * (m * self.heat_map - 1) + 1)

        self.velocity[2,0,:,:] = 0
        self.velocity[2,-1,:,:] = 0
        self.velocity[2,:,0,:] = 0
        self.velocity[2,:,-1,:] = 0
        self.velocity[:,:,:,0] = 0
        self.quantities_vapor[:,0,:] = self.quantities_vapor[:,-1,:]
        self.quantities_vapor[0,:,:] = self.quantities_vapor[-1,:,:]
        self.quantities_clouddrop[:,:,[0,-1]] = 0
        self.quantities_raindrop[:,:,[0,-1]] = 0
        self.quantities_clouddrop[:,[0,-1],:] = 0
        self.quantities_raindrop[:,[0,-1],:] = 0
        self.quantities_clouddrop[[0,-1],:,:] = 0
        self.quantities_raindrop[[0,-1],:,:] = 0
        print('boundary condition applied')
         
     def update_quantities(self):
         qvs = self.getSaturationRatio(self, 'pressure')
         qv = np.copy(self.quantities_vapor)
         qc = np.copy(self.quantities_clouddrop)
         qr = np.copy(self.quantities_raindrop)
         change = qvs - qv
         Ac = self.alpha_A * (qc - 1.0e-3)
         Kc = self.alpha_K * qc * qr
         Er = self.alpha_E * qc * np.sqrt(qr)
         for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.quantities_vapor[i,j,k] = qv + min(change[i,j,k], qc[i,j,k]) + Er[i,j,k]
                    self.quantities_clouddrop[i,j,k] = qc - min(change[i,j,k], qc[i,j,k]) - Ac[i,j,k] - Kc[i,j,k]
                    self.quantities_raindrop[i,j,k] = qr + Ac[i,j,k] + Kc[i,j,k] - Er[i,j,k]
         print('quantities updated')
    
     def update_temperature(self):
         X = np.zeros(self.shape)
         qvs = self.getSaturationRatio(self, 'pressure')
         cp = self.getHeatCapacity(self)
         for i in range(self.shape[0]):
              for j in range(self.shape[1]):
                   for k in range(self.shape[2]):
                        if (qvs[i,j,k] - self.quantities_vapor) < self.quantities_clouddrop:
                             X[i,j,k] = (qvs[i,j,k] - self.quantities_vapor[i,j,k]) / (self.quantities_sum_water[i,j,k] + self.quantities_dryair[i,j,k])
                        else:
                             X[i,j,k] = self.quantities_clouddrop[i,j,k] / (self.quantities_sum_water[i,j,k] + self.quantities_dryair[i,j,k])
         self.temperature += self.latenthead / cp * X
         print('temperature updated')

     def update_time(self):
         self.time += self.delta_t
         print('time updated')

     def step(self):
        velocity_0 = np.copy(self.velocity)

        # Advection is computed backwards in time as described in Stable Fluids.
        advection_map = self.indices - self.velocity
        # SciPy's spline filter introduces checkerboard divergence.
        # A linear blend of the filtered and unfiltered fields based
        # on some value epsilon eliminates this error.
        def advect(field, filter_epsilon=10e-2, mode='constant'):
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epsilon) + field * filter_epsilon
            return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)

        # Apply advection to each axis of the
        # velocity field and each user-defined quantity.
        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])


        
        self.vorticity_confinement(self)
        self.apply_b_external(self)

        # Compute the jacobian at each point in the
        # velocity field to extract curl and divergence.
        """ jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)

        divergence = jacobian.trace() """

        # If this curl calculation is extended to 3D, the y-axis value must be negated.
        # This corresponds to the coefficients of the levi-civita symbol in that dimension.
        # Higher dimensions do not have a vector -> scalar, or vector -> vector,
        # correspondence between velocity and curl due to differing isomorphisms
        # between exterior powers in dimensions != 2 or 3 respectively.
        #curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        #curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

        # Apply the pressure correction to the fluid's velocity field.
        pressure = self.pressure_solver(self.divergence(self.velocity).flatten()).reshape(self.shape)
        #pressure_solverは圧力の差分行列が仕込まれたfactorized関数。これに右辺を入れることで計算を行う。これは陰解法を解くのに有効な手段である。
        self.velocity -= np.gradient(pressure)

        setattr(self, 'quantities_clouddrop', advect(getattr(self, 'quantities_clouddrop')))
        setattr(self, 'quantities_vapor', advect(getattr(self, 'quantities_vapor')))
        setattr(self, 'quantities_raindrop', advect(getattr(self, 'quantities_raindrop')))
        setattr(self, 'temperature', advect(getattr(self, 'temperature')))

        self.update_quantities(self)
        self.update_temperature(self)
        self.boundary_condition(self)
        self.update_time(self)

        curl = self.rotate(self, 'velocity')
        div = self.divergence(self, 'velocity')

        return div, curl, pressure

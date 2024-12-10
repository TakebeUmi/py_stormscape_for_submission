#使用するライブラリのインポート
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
import time
import math
from noise import pnoise2
from enum import Enum
import sys
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from numerical import difference, operator

class AXIS:
    X = 1
    Y = 2
    Z = 3

class Fluid:
     def __init__(self, shape, *quantities, delta_x, epsilon, delta_t, z1, externalForce, phi_rel, gamma_heat, gamma_vapor, E):
        #グリッドとソルバーの作成
        self.shape = shape
        self.time = 0.0
        self.epsilon = epsilon
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.M_air = 28.96 * 1e-3 #kg/mol
        self.M_W = 18.02 * 1e-3 #kg/mol
        self.gamma = -0.0065 #K/m
        self.one_atm = np.zeros(shape)
        self.one_atm[:,:,:] = 101300
        #１気圧(＝101325パスカル)の設定
        self.E = E
        self.generalGasConstant = 8.31
        #J/molK
        self.T_ISA = 273
        self.gravity = 9.81
        #m/s2
        self.phi_rel = phi_rel
        self.gamma_heat = gamma_heat
        self.gamma_vapor = gamma_vapor
        self.alpha_A = 1.0e-2
        self.alpha_K = 1.0
        self.alpha_E = 1.0e-1
        self.nu = 1.71 * 1e-6 #m2/s
        #パーリンノイズをかけることで地表での水蒸気・温度分布にランダム性を持たせる
        self.vapor_map = np.zeros(shape[:2])
        self.heat_map = np.zeros(shape[:2])
        scale = 10.0
        for i in range(shape[0]):
             for j in range(shape[1]):
                  self.vapor_map[i,j] = (pnoise2(i / scale, j / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0) + 1)/2
                  self.heat_map[i,j] = (pnoise2(i / scale, j / scale, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0) + 1) / 2

        self.Rd = 287 #J/(kg K)
        self.latenthead = 2.5
        #J/kg
        MAX = max(max(self.shape[0], self.shape[1]), self.shape[2])
        self.a = self.delta_t*self.nu*MAX**3
        def f(z, z1):
             if z*delta_x <= z1:
                  return self.T_ISA + self.gamma * z * self.delta_x
             else:
                  return self.T_ISA + (2 * self.gamma * z1 - self.gamma * z * self.delta_x) 
        def Temperature_altitude(z1):
            z = np.zeros(shape)
            for k in range(shape[2]):
                 for i in range(shape[0]):
                     for j in range(shape[1]):
                        z[i,j,k] = f(k, z1)
            return z
        
        #高度に依存した温度分布を初期分布として設定
        self.temperature_altitude = Temperature_altitude(z1)
        self.T_air = Temperature_altitude(z1)
        self.temperature = self.T_air
        self.dimensions = len(shape)
        print('shape and dimension calculated')

        #外力項の設定
        self.externalForce = externalForce
        height_grid = np.arange(shape[2]).reshape(1, 1, shape[2])
        self.height_grid = np.broadcast_to(height_grid, shape) * delta_x

        #水蒸気密度の初期分布を設定
        for q in quantities:
            setattr(self, q, np.zeros(shape))
        initial_vapor_distribution = np.zeros(shape)
        for k in range(shape[2]):
             for i in range(shape[0]):
                 for j in range(shape[1]):
                    initial_vapor_distribution[i,j,k] = math.exp(-5.26e-4 * self.height_grid[i,j,k] + 2.30) 
        self.quantities_vapor = initial_vapor_distribution* 1e-3

        #大気中の水分の密度
        self.quantities_sum_water = self.quantities_clouddrop + self.quantities_raindrop + self.quantities_vapor
        #乾燥空気の密度
        self.quantities_dryair = np.full(shape, 1.293)
        #kg/m3

        #高度に圧力の分布を設定し、圧力の初期分布にする
        def pressure_altitude_field():
            p = (self.one_atm * (np.ones_like(self.height_grid) + self.gamma * self.height_grid / self.T_ISA) ** 5.2561)
            print(f"Max value of pressure_altitude: {np.max(p)}")
            return p
        self.pressure = pressure_altitude_field()
        self.pressure_0 = np.zeros(shape)



        #速度の初期分布の設定
        self.velocity = np.zeros((self.dimensions, *shape))
        self.velocity[2,shape[0]//2-shape[0]//8:shape[0]//2+shape[0]//8,shape[1]//2-shape[1]//8:shape[1]//2+shape[1]//8,0] = 2.0

        #移流計算に利用するもの
        self.indices = np.indices(shape) * self.delta_x
        laplacian = operator(shape, difference(2, 1))
        self.pressure_solver = factorized(laplacian)
        self.advect_order = 3        
        print('LU decomposed')
        

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
     #スカラー場qの位置(x,y,z)から値を持ってくる
    
     def get_offset(self, axis):
          offsets = {
        AXIS.X: (0, 0.5, 0.5),
        AXIS.Y: (0.5, 0, 0.5),
        AXIS.Z: (0.5, 0.5, 0)
          }
          return offsets.get(axis, (0, 0, 0))  # デフォルト値Vec3f(0, 0, 0)     
        #offsetの取得

     #以下補間の実装
     def get_interpolation(self, p, q):
         return self.trilinear_interp(p,q)
     
     def get_face_value(self, q, p, axis):
          offset = self.get_offset(axis)
          return self.get_interpolation(p-np.array(offset),q)
     
     def get_center_value(self,q,p):
         offset = (0.5,0.5,0.5)
         return self.get_interpolation(p-np.array(offset),q)
     
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

     def trace(self, orig, u):
         dir = np.zeros(self.dimensions)
         for i in range(self.dimensions):
             dir[i] = u[i] * self.delta_t / self.delta_x
         target = orig - dir
         return target #[a,b,c]
     
     #ヤコビ法
     def lin_solve(self, x, x0, a, c):                                            
          for i in range(1,self.shape[0]-1):
               for j in range(1,self.shape[1]-1):
                    for k in range(1,self.shape[2]-1):
                         x[i,j,k] = x0[i,j,k] + a * (x[i-1,j,k] + x[i+1,j,k] + x[i,j+1,k] + x[i,j-1,k] + x[i,j,k+1] + x[i,j,k-1]) / c
          return x
     
     def laplacian_matrix_3d(self):
        dx = self.delta_x
        derivative = 2
        diff_coefficients, _ = difference(derivative, accuracy=1)
        laplacian = operator(self.shape, (diff_coefficients / dx**2, _))
        return laplacian


     def diffuse(self, element):
        #  MAX = max(max(self.shape[0], self.shape[1]), self.shape[2])
        #  a = dt*diff*MAX**3
        #  return self.lin_solve(b, x, x0, x0, a, 1+6*a)
        laplacian = self.laplacian_matrix_3d()
        A = sp.identity(laplacian.shape[0]) - self.nu * laplacian
        N = np.prod(self.shape)
        b = element.ravel()
        x, info = cg(A, b, atol=1e-8)
        x = x.reshape(element.shape)
        return x
     
     def diffuse_velocity(self):
        vx = self.velocity[0]
        vy = self.velocity[1]
        vz = self.velocity[2]
        self.velocity[0] = self.diffuse(vx)
        self.velocity[1] = self.diffuse(vy)
        self.velocity[2] = self.diffuse(vz)
    
     def compute_divergence(self):
         div = np.zeros(self.shape)
         for i in range(self.dimensions):
             diff = (np.roll(self.velocity[i], -1, axis=i) - self.velocity[i]) / self.delta_x
             div += diff
         return div
     
     def correct_velocity(self):
         grad_p = np.gradient(self.pressure, self.delta_x, edge_order=2)
         for i in range(self.dimensions):
             self.velocity[i] -= grad_p[i]

     def pressure_projection(self):     
        diff_coefficients, points = difference(2, accuracy=1)
        laplacian = operator(self.shape, (diff_coefficients / self.delta_x**2, points))

        divergence = self.compute_divergence().flatten()
        A = laplacian
        b = -divergence
        pressure, info = cg(A, b, atol=1e-8)
        if info == 0:
            print("CG法は正常に収束しました")
        else:
            print("CG法は収束しませんでした。info =", info)
        
        self.correct_velocity()

     
        
     #速度の拡散の計算
    #  def diffuse_velocity(self):
    #      v0 = np.copy(self.velocity)
    #      v1 = np.copy(self.velocity)
    #      self.velocity[0] = self.lin_solve(v1[0], v0[0], self.a, 1+6*self.a)
    #      self.velocity[1] = self.lin_solve(v1[1], v0[1], self.a, 1+6*self.a)
    #      self.velocity[2] = self.lin_solve(v1[2], v0[2], self.a, 1+6*self.a)


     #スカラー量の移流の計算
     def advect_scalar_field(self):
        v1 = self.velocity
        qc = np.copy(self.quantities_clouddrop)
        qr = np.copy(self.quantities_raindrop)
        qv = np.copy(self.quantities_vapor)
        for i in range(self.shape[0]):
             for j in range(self.shape[1]):
                 for k in range(self.shape[2]):
                    center = (0.5+i, 0.5+j, 0.5+k)
                    v_orig = (self.get_face_value(v1[0], center, AXIS.X), self.get_face_value(v1[1], center, AXIS.Y), self.get_face_value(v1[2], center, AXIS.Z))
                    p = self.trace(center, v_orig)
                    qc[i,j,k] = self.get_center_value(qc,p)
                    qv[i,j,k] = self.get_center_value(qv,p)
                    qr[i,j,k] = self.get_center_value(qr,p)
        self.quantities_clouddrop = qc
        self.quantities_raindrop = qr
        self.quantities_vapor = qv
     
     
     #ベクトルの回転
     def rotate(self,vector_field):
            F = getattr(self, vector_field)
            Fx_y, Fx_z, Fx_x = np.gradient(F[0],self.delta_x)  # Fx成分のy, z, xに沿った勾配
            Fy_y, Fy_z, Fy_x = np.gradient(F[1],self.delta_x)  # Fy成分のy, z, xに沿った勾配
            Fz_y, Fz_z, Fz_x = np.gradient(F[2],self.delta_x)  # Fz成分のy, z, xに沿った勾配
            
            curl_x = Fz_y - Fy_z  # カールのx成分
            curl_y = Fx_z - Fz_x  # カールのy成分
            curl_z = Fy_x - Fx_y  # カールのz成分

            curl_F = np.stack((curl_x, curl_y, curl_z), axis=0)
            
            # jacobian_shape_rotate = (self.dimensions, ) * 2
            # partials_rotate = tuple(np.gradient(d) for d in getattr(self, vector_field))
            # jacobian_rotate = np.stack(partials_rotate).reshape(*jacobian_shape_rotate, *self.shape)
            # curl_mask = np.triu(np.ones(jacobian_shape_rotate, dtype=bool), k=1)
            # rotation = (jacobian_rotate[curl_mask] - jacobian_rotate[curl_mask.T]).squeeze()
            #print('rotation calculated')
            return curl_F
     
     #ベクトルの発散   
     def divergence(self, vector_field):
            jacobian_shape_div = (self.dimensions, ) * 2
            partials_div = tuple(np.gradient(d,self.delta_x) for d in getattr(self, vector_field))
            jacobian_div = np.stack(partials_div).reshape(*jacobian_shape_div, *self.shape)
            return jacobian_div.trace()
     
     #ベクトルの勾配
     def grad(self, scalar_field):
            g0, g1, g2 = np.gradient(scalar_field,self.delta_x)
            g = np.stack((g0, g1, g2), axis=0)
            return g
     
     #渦の付け加え(vorticity confinement)
     def vorticity_confinement(self):
            w = self.rotate('velocity')
            print(f"min value of w: {np.min(w)}")
            print(f"Max value of w: {np.max(w)}")
            k = self.grad(np.linalg.norm(w,axis=0))
            norm_k = np.linalg.norm(k, axis=0)
            norm_k[norm_k == 0] = 1
            N = k / norm_k
            transpose_N = np.transpose(N, (1,2,3,0))
            transpose_w = np.transpose(w, (1,2,3,0))
            vorticity_confinement = self.epsilon * self.delta_x * (np.transpose(np.cross(transpose_N, transpose_w), (3,0,1,2)))
            print(f"min value of vorticity confinement: {np.min(vorticity_confinement)}")
            print(f"Max value of vorticity confinement: {np.max(vorticity_confinement)}")
            self.velocity += self.delta_t * vorticity_confinement
     
     #高さに対する標準大気の圧力の計算
     def pressure_altitude_field(self):
        p = (self.one_atm * (np.ones_like(self.height_grid) + self.gamma * self.height_grid / self.T_ISA) ** 5.2561)
        print(f"Max value of pressure_altitude: {np.max(p)}")
        return p
     
     #モル分率の計算
     def getMoleFraction(self, quantity):
        self.quantities_sum_water = self.quantities_clouddrop + self.quantities_raindrop + self.quantities_raindrop
        q = getattr(self, quantity)
        X_i = q / (self.quantities_dryair + self.quantities_sum_water)
        return X_i
     
     #平均モル質量の計算
     def getAverageMolarMass(self):
        X_V = self.getMoleFraction('quantities_vapor')
        M_th = X_V * self.M_W + (1 - X_V) * self.M_air
        return M_th
     
     #isentropic exponentの計算
     def getIsentropicExponent(self):
        M_th = self.getAverageMolarMass()
        X_V = self.getMoleFraction('quantities_vapor')
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        return gammma_th
     
     #浮力の計算
     def compute_buoyancy(self):
        M_th = self.getAverageMolarMass()
        X_V = self.getMoleFraction('quantities_vapor')
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        T_th = np.zeros(self.shape)
        B = np.zeros((self.dimensions,*self.shape))
        T_th= self.temperature
        for i in range(gammma_th.shape[0]):
             for j in range(gammma_th.shape[1]):
                  for k in range(gammma_th.shape[2]):
                       
                       B[2,i,j,k] = self.gravity * (self.M_air * T_th[i,j,k] / (M_th[i,j,k] * self.T_air[i,j,k]) - 1)

        return B
     
     #浮力と外力による速度の更新
     def apply_b_external(self):
         b = self.compute_buoyancy()
         
         e = np.zeros((self.dimensions, *self.shape))
         for i in range(self.shape[0]):
             for j in range(self.shape[1]):
                 for k in range(self.shape[2]):
                     for l in range(self.dimensions):
                      e[l,i,j,k] = self.externalForce[l]
         self.velocity += self.delta_t * (b + e)
     
     #大気の熱容量分布の計算
     def getHeatCapacity(self):
         gamma_th = self.getAverageMolarMass()
         M_th = self.getAverageMolarMass()
         cp_th = np.zeros(self.shape)
         for i in range(cp_th.shape[0]):
             for j in range(cp_th.shape[1]):
                  for k in range(cp_th.shape[2]):
                       cp_th[i,j,k] = gamma_th[i,j,k] * self.generalGasConstant / (M_th[i,j,k] * (gamma_th[i,j,k] - 1))
         return cp_th
     
     #飽和水蒸気量の計算
     def getSaturationRatio(self, pressure):
         p = getattr(self, pressure)
         T = self.temperature - 273
         qvs = np.zeros(self.shape)
         for i in range(qvs.shape[0]):
            for j in range(qvs.shape[0]):
                for k in range(qvs.shape[0]):
                    qvs[i,j,k] = (380.16 / p[i,j,k] )* math.exp(17.67 * T[i,j,k] / (T[i,j,k] + 243.50))
         return qvs
     
     #境界条件の設定
     def boundary_condition(self):
        m = 2
        self.quantities_vapor[:,:,0] = self.phi_rel * self.getSaturationRatio('one_atm')[:,:,0] * (self.gamma_vapor * (m * self.vapor_map - 1) + 1)
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

        self.quantities_clouddrop[self.quantities_clouddrop < 0] = 0
        self.quantities_raindrop[self.quantities_raindrop < 0] = 0
        self.quantities_vapor[self.quantities_vapor < 0] = 0
        self.pressure[self.pressure < 0] = 0

     #水蒸気・雲粒・雨粒の相互変化    
     def update_quantities(self): 
         qvs = self.getSaturationRatio('pressure')
         qv = np.copy(self.quantities_vapor)
         qc = np.copy(self.quantities_clouddrop)
         qr = np.copy(self.quantities_raindrop)
         change = qvs - qv
         Ac = self.alpha_A * (qc - 1.0e-3)
         Kc = self.alpha_K * qc * qr
         Er = self.alpha_E * qc * np.sqrt(qr)

         
         print(f"Max value of qvs: {np.max(qvs)}")
         for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.quantities_vapor[i,j,k] = qv[i,j,k] + min(change[i,j,k], qc[i,j,k]) + Er[i,j,k]
                    self.quantities_clouddrop[i,j,k] = qc[i,j,k] - min(change[i,j,k], qc[i,j,k]) - Ac[i,j,k] - Kc[i,j,k]
                    self.quantities_raindrop[i,j,k] = qr[i,j,k] + Ac[i,j,k] + Kc[i,j,k] - Er[i,j,k]
     #温度場の更新
     def update_temperature(self):
         X = np.zeros(self.shape)
         qvs = self.getSaturationRatio('pressure')
         cp = self.getHeatCapacity()
         for i in range(self.shape[0]):
              for j in range(self.shape[1]):
                   for k in range(self.shape[2]):
                        if (qvs[i,j,k] - self.quantities_vapor[i,j,k]) < self.quantities_clouddrop[i,j,k]:
                             X[i,j,k] = (qvs[i,j,k] - self.quantities_vapor[i,j,k]) / (self.quantities_sum_water[i,j,k] + self.quantities_dryair[i,j,k])
                        else:
                             X[i,j,k] = self.quantities_clouddrop[i,j,k] / (self.quantities_sum_water[i,j,k] + self.quantities_dryair[i,j,k])
             
         self.temperature += (self.latenthead / cp )* X

     def update_time(self):
         self.time += self.delta_t

     def step(self):
        #移流の定義
        advection_map = (self.indices*self.delta_x - self.velocity)
        def advect(field, filter_epsilon=1e-2, mode='nearest'):
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epsilon) + field * filter_epsilon
            return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)



        #1.移流
        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])
        self.boundary_condition()
        #2.拡散
        self.diffuse_velocity()
        self.boundary_condition()
        #3,4,5.渦の付け加え
        self.vorticity_confinement()
        self.boundary_condition()
        #6,7,8.外力と浮力による速度の更新
        self.apply_b_external()
        self.boundary_condition()
        #9,10.投影
        # pressure = self.pressure_solver(self.divergence('velocity').flatten()).reshape(self.shape) * 1e-3
        # self.pressure += pressure
        self.pressure_projection()
        print(f"Max value of pressure(after boundary): {np.max(self.pressure)}")
        print(f"Max value of clouddrop: {np.max(self.quantities_clouddrop)}")

        # self.velocity -= np.gradient(self.pressure)
        #11.スカラー場の移流
        self.advect_scalar_field()
        setattr(self, 'temperature', advect(getattr(self, 'temperature')))
        self.boundary_condition()
        #12,13.水蒸気・雲粒・雨粒の相互変化
        self.update_quantities()
        self.boundary_condition()
        #14,15,16,17,18.温度場の更新
        self.update_temperature()
        self.boundary_condition()
        #19
        self.update_time()

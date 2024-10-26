import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
import time
import math

from numerical import difference, operator


class Fluid:
    def __init__(self, shape, *quantities, pressure_order=1, advect_order=3, delta_x, epsilon, delta_t, z1, externalForce, phi_rel, gamma_heat, gamma_vapor):
        #グリッドとソルバーの作成
        self.shape = shape
        self.epsilon = epsilon
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.M_air = 28.96
        self.M_W = 18.02
        self.gamma = 0.0065
        self.one_atm = 101300
        #１気圧＝101325パスカル
        self.generalGasConstant = 8.31
        #J/molK
        self.T_ISA = 273
        self.gravity = 9.81
        #m/s2
        self.phi_rel = phi_rel
        self.gamma_heat = gamma_heat
        self.gamma_vapor = gamma_vapor

        self.theta = np.zeros(shape)
        self.pressure = np.zeros(shape)
        self.pressure_0 = np.zeros(shape)
        self.temperature = np.zeros(shape)
        self.temperature_0 = np.zeros(shape)
        self.Rd = 287
        def f(z, z1):
             if z <= z1:
                  return self.T_ISA + self.gamma * z / self.delta_x
             else:
                  return self.T_ISA + (2 * self.gamma * z1 - self.gamma * z) / self.delta_x
        def Temperture_altitude(z1):
            z = np.arrange(shape[2])
            z = z[None, None, :]
            for k in range(z1):
                 z[:, :, k] = f(k, z1)
            return z
        
        self.T_air = Temperture_altitude(z1)
        #ここ高度とz1の兼ね合いがあいまいなので調整
        self.dimensions = len(shape)
        print('shape and dimension calculated')
        # Prototyping is simplified by dynamically 
        # creating advected quantities as needed.
        self.quantities_clouddrop = quantities[0]
        self.quantities_raindrop =quantities[1]
        self.quantities_vapor = quantities[2]
        self.quantities_dryair = 1.293
        #kg/m3
        self.externalForce = externalForce
        height_grid = np.arange(10).reshape(10, 1, 1)
        # それを10x10の平面全体にブロードキャストして、高さの配列を生成
        self.height_grid = np.broadcast_to(height_grid, shape) * delta_x
        #高さはkm表示。そのためにdelta_xで調整。必要であれば追加

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
    

    def rotate(self,vector_field):
            jacobian_shape_rotate = (self.dimensions, ) * 2
            partials_rotate = tuple(np.gradient(d) for d in getattr(self, vector_field))
            jacobian_rotate = np.stack(partials_rotate).reshape(*jacobian_shape_rotate, *self.shape)
            curl_mask = np.triu(np.ones(jacobian_shape_rotate, dtype=bool), k=1)
            rotation = (jacobian_rotate[curl_mask] - jacobian_rotate[curl_mask.T]).squeeze()
            return rotation
        
    def divergence(self, vector_field):
            jacobian_shape_div = (self.dimensions, ) * 2
            jacobian_div = np.stack(partials_div).reshape(*jacobian_shape_div, *self.shape)
            partials_div = tuple(np.gradient(d) for d in getattr(self, vector_field))
            return jacobian_div.trace()

    def grad(self, scalar_field):
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
    
    def pressure_altitude_field(self):
        return ((self.T_ISA * np.ones_like(self.height_grid) - self.gammma * self.height_grid) ** 5.2561)

    def getMoleFraction(self):
        q_w = self.quantities_clouddrop + self.quantities_raindrop + self.quantities_raindrop
        X_V = (q_w) / (self.quantities_dryair + q_w)
        return X_V

    def getAverageMolarMass(self):
        X_V = self.getMoleFraction(self)
        M_th = X_V * self.M_W + (1 - X_V) * self.M_air
        return M_th

    def getIsentropicExponent(self):
        M_th = self.getAverageMolarMass(self)
        X_V = self.getMoleFraction(self)
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        return gammma_th
    
    def compute_buoyancy(self):
        M_th = self.getAverageMolarMass(self)
        X_V = self.getMoleFraction(self)
        Y_V = X_V * self.M_W / M_th
        gammma_th = Y_V * 1.33 + (np.ones_like(Y_V) - Y_V) * 1.4
        T_th = np.zeros(self.shape)
        B = np.zeros((self.dimensions,*self.shape))
        #M_air...スカラー値
        #M_th...スカラー場
        #T_th...スカラー場
        #T_air...スカラー場
        #配列同士の積は求められないので、要素ごとに計算
        for i in len(gammma_th.shape[0]):
             for j in len(gammma_th.shape[1]):
                  for k in len(gammma_th.shape[2]):
                       T_th[i,j,k] = self.pressure_altitude_field()[i,j,k] ** gammma_th[i,j,k]
                       B[2,i,j,k] = self.gravity * (self.M_air * T_th[i,j,k] / (M_th[i,j,k] * self.T_air[i,j,k]))
        
        return B
    
    def apply_b_external(self):
         b = self.compute_buoyancy
         e = self.externalForce
         self.velocity += self.delta_t * (b + e)

    def getHeatCpacity(self):
         gamma_th = self.getAverageMolarMass(self)
         M_th = self.getAverageMolarMass(self)
         cp_th = np.zeros(self.shape)
         for i in len(cp_th.shape[0]):
             for j in len(cp_th.shape[1]):
                  for k in len(cp_th.shape[2]):
                       cp_th[i,j,k] = gamma_th[i,j,k] * self.generalGasConstant / (M_th[i,j,k] * (gamma_th[i,j,k] - 1))
         return cp_th
    
    def potential_temperture(self):
         p0 = self.pressure_0
         p1 = self.pressure
         kappa = self.Rd / self.getHeatCpacity(self)
         T0 = self.temperture_0
         theta = np.zeros(self.shape)
         for i in len(theta.shape[0]):
              for j in len(theta.shape[1]):
                   for k in len(theta.shape[2]):
                        theta[i,j,k] = ((p1[i,j,k]/p0[i,j,k]) ** kappa[i,j,k]) * T0
         return theta

    def getSaturationRatio(self, pressure):
         p = getattr(self, pressure)
         T = self.temperature
         qvs = np.zeros(self.shape)
         for i in len(qvs.shape[0]):
            for j in len(qvs.shape[0]):
                for k in len(qvs.shape[0]):
                    qvs[i,j,k] = 380.16 / p[i,j,k] * math.exp(17.67 * T[i,j,k] / (T[i,j,k] + 243.50))
         return qvs
         
    def update_quantities(self):
         qvs = self.getSaturationRatio(self, 'pressure')
         qv = np.copy(self.quantities_vapor)
         qc = np.copy(self.quantities_clouddrop)
         qr = np.copy(self.quantities_raindrop)
         change = qvs - qv
         for i in len(self.shape[0]):
            for j in len(self.shape[1]):
                for k in len(self.shape[2]):
                    self.quantities_vapor[i,j,k] = qv + min(change[i,j,k], qc[i,j,k])
                    self.quantities_clouddrop[i,j,k] = qc - min(change[i,j,k], qc[i,j,k])
                    self.quantities_raindrop[i,j,k] = qr 

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

        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))

        curl = self.rotate(self, 'velocity')
        div = self.divergence(self, 'velocity')
        return div, curl, pressure

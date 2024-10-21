import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
import time

from numerical import difference, operator


class Fluid:
    def __init__(self, shape, *quantities_vapor, *quantities_clouddrop, *quantities_raindrop, pressure_order=1, advect_order=3):
        self.shape = shape
        self.dimensions = len(shape)
        print('shape and dimension calculated')
        # Prototyping is simplified by dynamically 
        # creating advected quantities as needed.
        self.quantities_clouddrop = quantities_clouddrop
        self.quantities_raindrop = quantities_raindrop
        self.quantities_vapor = quantities_vapor
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
        self.vapor_quantities = np.zeros(self.dimensions)
        self.raindrop_quantities = np.zeros(self.dimensions)
        self.clouddrop_quantities = np.zeros(self.dimensions)

        laplacian = operator(shape, difference(2, pressure_order))
        rotate = operator(shape, )
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

    def step(self):
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

        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))

        # Compute the jacobian at each point in the
        # velocity field to extract curl and divergence.
        jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)

        divergence = jacobian.trace()

        # If this curl calculation is extended to 3D, the y-axis value must be negated.
        # This corresponds to the coefficients of the levi-civita symbol in that dimension.
        # Higher dimensions do not have a vector -> scalar, or vector -> vector,
        # correspondence between velocity and curl due to differing isomorphisms
        # between exterior powers in dimensions != 2 or 3 respectively.
        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

        # Apply the pressure correction to the fluid's velocity field.
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        #pressure_solverは圧力の差分行列が仕込まれたfactorized関数。これに右辺を入れることで計算を行う。これは陰解法を解くのに有効な手段である。
        self.velocity -= np.gradient(pressure)

        return divergence, curl, pressure

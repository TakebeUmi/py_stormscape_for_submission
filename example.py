###使うライブラリの選択
import numpy as np

from scipy.special import erf

from fluid import Fluid
import pyopenvdb as vdb
import os

###使用するグリッドのサイズとユーザーが設定するパラメータ群
xy = 30
z = 30
RESOLUTION = xy, xy, z
DURATION = 50
delta_x = 50 #m 
epsilon = 0.25e-2
delta_t = 60 #s
z1 = 8000 #m
externalForce = [0.0, 0.0, 0.0]
phi_rel = 1.0
gamma_heat = 1.27
gamma_vapor = 0.13
E = 0.5
quantities = ('quantities_clouddrop','quantities_raindrop','quantities_vapor')

# Fluidインスタンスの作成
cloud = Fluid(
    RESOLUTION, *quantities,
    delta_x=delta_x,
    epsilon=epsilon,
    delta_t=delta_t,
    z1=z1,
    externalForce=externalForce,
    phi_rel=phi_rel,
    gamma_heat=gamma_heat,
    gamma_vapor=gamma_vapor,
    E=E
)

print('class calculated')

###DURATIONの回数だけ計算を反復する
x = cloud.shape[0]
for f in range(DURATION):
    print(f'Computing frame {f + 1} of {DURATION}.')
    cloud.step()
    #1タイムステップでの計算

    #####以下は書き出し

    vdb_grid = vdb.FloatGrid()
    vdb_grid_100 = vdb.FloatGrid()
    vdb_grid.copyFromArray(cloud.quantities_clouddrop * 10)
    vdb_grid_100.copyFromArray(cloud.quantities_clouddrop * 100)
    output_dir = f"output/output_cube_fluid_{x}"
    output_dir_100 = f"output/output_cube_fluid_100times_{x}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_100, exist_ok=True)
    file_name = f"{output_dir}/output_cube_fluid_{x}_frame_{f:04d}.vdb"
    file_name_100 = f"{output_dir_100}/output_cube_fluid_100times_{x}_frame_{f:04d}.vdb"
    vdb.write(file_name, grids=[vdb_grid])
    vdb.write(file_name_100, grids=[vdb_grid_100])
    file_name = 'density_memo.txt'
    # np.savetxt('quantities_clouddrop.txt',cloud.quantities_clouddrop[:,cloud.shape[0]//2,:])
    # np.savetxt('quantities_vapor.txt',cloud.quantities_vapor[:,cloud.shape[0]//2,:])
    with open('quantities_clouddrop.txt', 'a') as f:
        f.write(f"# Time step {cloud.time}\n")
        np.savetxt(f, cloud.quantities_clouddrop[:,cloud.shape[0]//2,:], fmt='%.6f')
    with open('quantities_vapor.txt', 'a') as f:
        f.write(f"# Time step {cloud.time}\n")
        np.savetxt(f, cloud.quantities_vapor[:,cloud.shape[0]//2,:], fmt='%.6f')

print('Saving simulation result.')


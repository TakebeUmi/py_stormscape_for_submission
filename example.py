###使うライブラリの選択
import numpy as np

from scipy.special import erf

from fluid import Fluid
import pyopenvdb as vdb
from PIL import Image
import os

###定数の具体的な値
xy = 50
z = 50
RESOLUTION = xy, xy, z
DURATION = 100
#今は適当な値を入れてるので後で値入れる
EPSILON = 1e-5
DELTA_X = 0.1

# INFLOW_PADDING = 40
# INFLOW_DURATION = 60
# INFLOW_RADIUS = 8
# INFLOW_VELOCITY = 1
# INFLOW_COUNT = 5
# 必須の引数
#shape, *quantities, delta_x, epsilon, delta_t, z1, externalForce, phi_rel, gamma_heat, gamma_vapor, E
delta_x = 50 #m 全長3840m * 1520m * 3840m
epsilon = 0.25e-2
delta_t = 60 #s
z1 = 8000 #m
externalForce = [0.0, 0.0, 0.005]
phi_rel = 1.0
gamma_heat = 0.
gamma_vapor = 0.
E = 0.5
#E,gamma_heat,gamma_vapor,phi_relが形状を決定する
# オプションの引数 quantities のみ指定し、pressure_order と advect_order を省略
quantities = ('quantities_clouddrop','quantities_raindrop','quantities_vapor')

# Fluidインスタンスの作成（pressure_order と advect_order は省略）
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
###Fluidインスタンスの作成
#1.shape 2.quantities 3.
#print('Generating fluid solver, this may take some time.')
#fluid = Fluid(RESOLUTION, 'dye')

#center = np.floor_divide(RESOLUTION, 2)
#r = np.min(center) - INFLOW_PADDING

#points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
#points = tuple(np.array((0, np.cos(p), np.sin(p))) for p in points)
#3dにするにあたってここを改善
#normals = tuple(-p for p in points)
#points = tuple(r * p + center for p in points)

###計算条件の設定

print('class calculated')
# inflow_velocity = np.zeros_like(fluid.velocity)
# inflow_dye = np.zeros(fluid.shape)
# """ for p, n in zip(points, normals):
#     mask = np.linalg.norm(fluid.indices - p[:, None, None, None], axis=0) <= INFLOW_RADIUS
#     inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY#nの方向に流入をかけて
#     inflow_dye[mask] = 1 """
#     #pは流入点、nは法線で速度の方向を表す
# half_minus = RESOLUTION[1]//2 - RESOLUTION[1]//20
# half_plus = RESOLUTION[1]//2 + RESOLUTION[1]//20
# bottom = RESOLUTION[2] //10
# inflow_velocity[2, half_minus:half_plus,half_minus:half_plus,:bottom] = INFLOW_VELOCITY
# inflow_dye[half_minus:half_plus,half_minus:half_plus,:bottom] = 1.0
#print('animation_calculating')
#frames = []

###DURATIONの回数だけ計算を反復する
x = cloud.shape[0]
for f in range(DURATION):
    print(f'Computing frame {f + 1} of {DURATION}.')
    # if f <= INFLOW_DURATION:
    #     fluid.velocity += inflow_velocity
    #     fluid.dye += inflow_dye
    #ctrl+/で一括コメントアウト
    

    cloud.step()
    # Using the error function to make the contrast a bit higher. 
    # Any other sigmoid function e.g. smoothstep would work.
    # curl = (erf(curl * 2) + 1) / 4
    # print(curl.shape)
    # print(fluid.shape)
    # print(fluid.dye.shape)
    # curl = np.linalg.norm(curl, axis=0)
    # print(curl.shape)
    #curlはそのままでは3*20*20*20の３次元ベクトル量。２次元の時はベクトルの大きさを取ってスカラー量で出力していたのでdstackの処理で詰まることはなかったが、３次元の場合はnormを取って20*20*20にする
    # color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))

    #####以下は書き出し

    vdb_grid = vdb.FloatGrid()
    vdb_grid_100 = vdb.FloatGrid()
    vdb_grid.copyFromArray(cloud.quantities_clouddrop * 10)
    vdb_grid_100.copyFromArray(cloud.quantities_clouddrop * 100)
    #課題：fluid.dyeの描画。np.arrayからopenvdbへ
    output_dir = f"output_cube_fluid_{x}"
    output_dir_100 = f"output_cube_fluid_100times_{x}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_100, exist_ok=True)
    file_name = f"{output_dir}/output_cube_fluid_{x}_frame_{f:04d}.vdb"
    file_name_100 = f"{output_dir}/output_cube_fluid_100times_{x}_frame_{f:04d}.vdb"
    vdb.write(file_name, grids=[vdb_grid])
    file_name = 'density_memo.txt'
    with open(file_name, "a") as file:
            file.write(f"{np.max(cloud.quantities_clouddrop)}\n")
    #print('color_shape=',color.shape)
    #color = (np.clip(color, 0, 1) * 255).astype('uint8')
    #frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))

print('Saving simulation result.')
file_name = 'density_memo.txt'
with open(file_name, "a") as file:
            file.write(f"\n")

#frames[0].save('example3d.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)
#課題：100*100*100のグリッドの10*10*10の領域で速度１を供給する

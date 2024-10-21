import numpy as np

from scipy.special import erf

from fluid import Fluid
import pyopenvdb as vdb
from PIL import Image
import os

xyz = 30
RESOLUTION =(xyz, ) * 3
DURATION = 200

INFLOW_PADDING = 40
INFLOW_DURATION = 60
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 1
INFLOW_COUNT = 5

print('Generating fluid solver, this may take some time.')
fluid = Fluid(RESOLUTION, 'dye')

#center = np.floor_divide(RESOLUTION, 2)
#r = np.min(center) - INFLOW_PADDING

#points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
#points = tuple(np.array((0, np.cos(p), np.sin(p))) for p in points)
#3dにするにあたってここを改善
#normals = tuple(-p for p in points)
#points = tuple(r * p + center for p in points)
print('class calculated')
inflow_velocity = np.zeros_like(fluid.velocity)
inflow_dye = np.zeros(fluid.shape)
""" for p, n in zip(points, normals):
    mask = np.linalg.norm(fluid.indices - p[:, None, None, None], axis=0) <= INFLOW_RADIUS
    inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY#nの方向に流入をかけて
    inflow_dye[mask] = 1 """
    #pは流入点、nは法線で速度の方向を表す
half_minus = RESOLUTION[1]//2 - RESOLUTION[1]//20
half_plus = RESOLUTION[1]//2 + RESOLUTION[1]//20
bottom = RESOLUTION[2] //10
inflow_velocity[2, half_minus:half_plus,half_minus:half_plus,:bottom] = INFLOW_VELOCITY
inflow_dye[half_minus:half_plus,half_minus:half_plus,:bottom] = 1.0
print('animation_calculating')
#frames = []
for f in range(DURATION):
    print(f'Computing frame {f + 1} of {DURATION}.')
    if f <= INFLOW_DURATION:
        fluid.velocity += inflow_velocity
        fluid.dye += inflow_dye
    

    curl = fluid.step()[1]
    # Using the error function to make the contrast a bit higher. 
    # Any other sigmoid function e.g. smoothstep would work.
    curl = (erf(curl * 2) + 1) / 4
    print(curl.shape)
    print(fluid.shape)
    print(fluid.dye.shape)
    curl = np.linalg.norm(curl, axis=0)
    print(curl.shape)
    #curlはそのままでは3*20*20*20の３次元ベクトル量。２次元の時はベクトルの大きさを取ってスカラー量で出力していたのでdstackの処理で詰まることはなかったが、３次元の場合はnormを取って20*20*20にする
    color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))
    dye_grid = vdb.FloatGrid()
    dye_grid.copyFromArray(fluid.dye)
    #課題：fluid.dyeの描画。np.arrayからopenvdbへ
    x = fluid.shape[0]
    output_dir = f"output_cube_fluid_{x}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/output_cube_fluid_{x}_frame_{f:04d}.vdb"
    vdb.write(file_name, grids=[dye_grid])
    print('color_shape=',color.shape)
    #課題：ここのcurlを3次元配列にする（今は3*20*20*20の4次元）
    color = (np.clip(color, 0, 1) * 255).astype('uint8')
    #frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))

print('Saving simulation result.')
#frames[0].save('example3d.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)
#課題：100*100*100のグリッドの10*10*10の領域で速度１を供給する

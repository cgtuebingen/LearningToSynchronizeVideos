import dijkstra
import numpy as np
import cv2
import re
import time

dir(dijkstra)


def pfm_load(filename):
  with open(filename) as file:
    rgb = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
      rgb = True
    elif header == 'Pf':
      rgb = False
    else:
      raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
      width, height = map(int, dim_match.groups())
    else:
      raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
      endian = '<'
      scale = -scale
    else:
      endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if rgb else (height, width)
    return np.reshape(data, shape)


def advanced_dijkstra(cost):
  assert len(cost.shape) == 2, "Shape should be 2-dim of cost matrix"
  cost -= cost.min()
  cost /= cost.max()
  cost = np.ascontiguousarray(cost)
  duration = time.time()
  [tour, box] = dijkstra.advanced_dijkstra(250, 30, cost)
  print(len(tour))
  duration = time.time() - duration
  print('function [advanced_dijkstra] finished in {} ms'.format(
      int(duration * 1000)))
  return np.reshape(tour, [-1, 2]), np.reshape(box, [-1, 4])


def plain_dijkstra(cost):
  assert len(cost.shape) == 2, "Shape should be 2-dim of cost matrix"
  cost -= cost.min()
  cost /= cost.max()
  cost[0, 0] = 0
  cost[-1, -1] = 0
  cost = np.ascontiguousarray(cost)
  duration = time.time()
  tour = dijkstra.plain_dijkstra(250, 30, cost)
  duration = time.time() - duration
  print('function [plain_dijkstra] finished in {} ms'.format(
      int(duration * 1000)))
  return np.reshape(tour, [-1, 2])


def accumulate_costs(cost2):
  cost = np.copy(cost2)
  assert len(cost.shape) == 2, "Shape should be 2-dim of cost matrix"
  cost -= cost.min()
  cost /= cost.max()
  cost = np.ascontiguousarray(cost)
  duration = time.time()
  dijkstra.accumulate(cost)
  duration = time.time() - duration
  print('function [plain_dijkstra] finished in {} ms'.format(
      int(duration * 1000)))
  return cost


cost = "/graphics/projects/scratch/wieschol/sync-corr/iter1/found/2012-06-22_L0004_2012-07-16_L0004_dcost.jpg"
#cost = "/home/vroni/masterarbeit/testData/costs_tours/coll_traffic_light_scene:traffic_light_left__coll_traffic_light_scene:traffic_light_right.png"
cost = cv2.imread(cost)[:, :, 0].astype("float32")
cost[0, 0] = 1

startTime = time.time()
[tour, box] = advanced_dijkstra(cost)


thickness = 1
img = np.copy(cost)
img = np.dstack([img, img, img])
img -= img.min()
img = img / img.max() * 255
print(tour[0, ...])


c = accumulate_costs(cost)
c -= c.min()
c /= c.max()
cv2.imwrite("accumulated.jpg", c * 255.)

# -------------------------------------------------------------------------
# show tour
# -------------------------------------------------------------------------
print(",,", len(tour))


def show_tour(tour, img):
  for r, s in tour:
    vs = max(r, 0)
    ve = min(r + thickness, img.shape[0])
    hs = max(s, 0)
    he = min(s + thickness, img.shape[1])
    img[vs:ve, hs:he, 0] = 0
    img[vs:ve, hs:he, 1] = 0
    img[vs:ve, hs:he, 2] = 255
  return img


img = show_tour(tour, img)

# -------------------------------------------------------------------------
# show stripes
# -------------------------------------------------------------------------
for i in range(0, img.shape[1] - 10, 250):
  print(i)
  img[:, i:i + 10, :] = 255

cv2.imwrite("out.jpg", img)

# -------------------------------------------------------------------------
# show finer boxes
# -------------------------------------------------------------------------
print(box.shape)
s = 0
for rrb, rre, ssb, sse in box:
  s += 1
  print(rrb, rre, ssb, sse)
  img[rrb: rre, ssb: sse, 1] = 255


cv2.imwrite("debug.jpg", img)

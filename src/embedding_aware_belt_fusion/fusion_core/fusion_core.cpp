// Late-fusion core of tools/eval_advercpm_fusion.py in C++.
//
// Same operations as the Python path, which follows OpenCOOD:
//   project       box_utils.project_world_objects + corner_to_center + boxes_to_corners_3d (float32 result)
//   fuse_and_score remove_large_pred_bbx, remove_bbx_abnormal_z, rotated NMS (fast_eval.nms_rotated_fast),
//                 GT-range mask, greedy TP/FP at several IoU thresholds (fast_eval.tp_fp_fast)
// BEV IoU is computed exactly by convex polygon clipping (Sutherland-Hodgman) in double precision and cast to
// float32 before any threshold comparison, as fast_eval does with shapely.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace py = pybind11;
using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using f64arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kPi = 3.14159265358979323846;
// Box filters and evaluation range. Defaults are OpenCOOD's (GT_RANGE x/y, z window [-3, 1], extent <= 6 m) and
// reproduce the CARLA results; set_filters() changes them for other sensor setups (e.g. roadside units ~7 m up).
float kRangeX = 140.f, kRangeY = 40.f;  // opencood.data_utils.datasets.GT_RANGE
float kZMin = -3.f, kZMax = 1.f, kMaxExtent = 6.f;

void x_to_world(const double* p, double m[4][4]) {
  // opencood.utils.transformation_utils.x_to_world; pose = [x, y, z, roll, yaw, pitch] in degrees
  const double cy = std::cos(p[4] * (kPi / 180.0)), sy = std::sin(p[4] * (kPi / 180.0));
  const double cr = std::cos(p[3] * (kPi / 180.0)), sr = std::sin(p[3] * (kPi / 180.0));
  const double cp = std::cos(p[5] * (kPi / 180.0)), sp = std::sin(p[5] * (kPi / 180.0));
  m[0][0] = cp * cy; m[0][1] = cy * sp * sr - sy * cr; m[0][2] = -cy * sp * cr - sy * sr; m[0][3] = p[0];
  m[1][0] = sy * cp; m[1][1] = sy * sp * sr + cy * cr; m[1][2] = -sy * sp * cr + cy * sr; m[1][3] = p[1];
  m[2][0] = sp;      m[2][1] = -cp * sr;               m[2][2] = cp * cr;                m[2][3] = p[2];
  m[3][0] = 0;       m[3][1] = 0;                      m[3][2] = 0;                      m[3][3] = 1;
}

struct Poly {
  double x[16], y[16];  // a quadrilateral clipped by four half-planes has at most 8 vertices
  int n = 0;
};

double area(const Poly& p) {
  double a = 0;
  for (int i = 0; i < p.n; ++i) {
    const int j = (i + 1) % p.n;
    a += p.x[i] * p.y[j] - p.x[j] * p.y[i];
  }
  return 0.5 * a;
}

// Clip convex polygon `s` (counter-clockwise) by the half-plane left of edge a->b.
Poly clip(const Poly& s, double ax, double ay, double bx, double by) {
  Poly out;
  if (s.n == 0) return out;
  auto side = [&](double px, double py) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); };
  for (int i = 0; i < s.n; ++i) {
    const int j = (i + 1) % s.n;
    const double si = side(s.x[i], s.y[i]), sj = side(s.x[j], s.y[j]);
    if (si >= 0) { out.x[out.n] = s.x[i]; out.y[out.n] = s.y[i]; ++out.n; }
    if ((si >= 0) != (sj >= 0)) {
      const double t = si / (si - sj);
      out.x[out.n] = s.x[i] + t * (s.x[j] - s.x[i]);
      out.y[out.n] = s.y[i] + t * (s.y[j] - s.y[i]);
      ++out.n;
    }
  }
  return out;
}

struct Box2 {
  Poly p;       // counter-clockwise
  double a;     // area
  double cx, cy, r;  // bounding circle
};

Box2 make_box(const float* c) {  // c: 8x3 corners, the first four (x, y) form the BEV polygon
  Box2 b;
  b.p.n = 4;
  for (int k = 0; k < 4; ++k) { b.p.x[k] = c[3 * k]; b.p.y[k] = c[3 * k + 1]; }
  double a = area(b.p);
  if (a < 0) {  // make counter-clockwise
    std::reverse(b.p.x, b.p.x + 4);
    std::reverse(b.p.y, b.p.y + 4);
    a = -a;
  }
  b.a = a;
  b.cx = (b.p.x[0] + b.p.x[1] + b.p.x[2] + b.p.x[3]) / 4;
  b.cy = (b.p.y[0] + b.p.y[1] + b.p.y[2] + b.p.y[3]) / 4;
  b.r = 0;
  for (int k = 0; k < 4; ++k) b.r = std::max(b.r, std::hypot(b.p.x[k] - b.cx, b.p.y[k] - b.cy));
  return b;
}

float iou(const Box2& a, const Box2& b) {
  const double d = std::hypot(a.cx - b.cx, a.cy - b.cy);
  const bool degenerate = a.a == 0 || b.a == 0;
  if (!degenerate && d > (a.r + b.r) * (1 + 1e-9) + 1e-9) return 0.f;
  double inter = 0;
  if (!degenerate) {
    Poly s = a.p;
    for (int k = 0; k < 4 && s.n > 0; ++k) {
      const int j = (k + 1) % 4;
      s = clip(s, b.p.x[k], b.p.y[k], b.p.x[j], b.p.y[j]);
    }
    inter = s.n >= 3 ? std::fabs(area(s)) : 0.0;
  }
  const double uni = a.a + b.a - inter;
  return static_cast<float>(inter / uni);  // 0/0 -> NaN for two degenerate boxes, as with shapely
}

}  // namespace

// World boxes (N, 12) = [location, center, extent, angle(roll, yaw, pitch)] -> ego-frame corners (N, 8, 3) float32.
f32arr project(f64arr boxes, f64arr world_to_ego, bool hwl) {
  const auto B = boxes.unchecked<2>();
  const auto W = world_to_ego.unchecked<2>();
  const py::ssize_t n = B.shape(0);
  f32arr out({n, py::ssize_t(8), py::ssize_t(3)});
  float* o = out.mutable_data();
  static const double sgn[8][3] = {{1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, -1},
                                   {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},  {-1, -1, 1}};
  for (py::ssize_t i = 0; i < n; ++i) {
    const double pose[6] = {B(i, 0) + B(i, 3), B(i, 1) + B(i, 4), B(i, 2) + B(i, 5), B(i, 9), B(i, 10), B(i, 11)};
    double m[4][4], t[4][4];
    x_to_world(pose, m);
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += W(r, k) * m[k][c];
        t[r][c] = s;
      }
    double cn[8][3];  // create_bbx(extent) corners in the ego frame
    for (int k = 0; k < 8; ++k) {
      const double v[3] = {sgn[k][0] * B(i, 6), sgn[k][1] * B(i, 7), sgn[k][2] * B(i, 8)};
      for (int r = 0; r < 3; ++r) cn[k][r] = t[r][0] * v[0] + t[r][1] * v[1] + t[r][2] * v[2] + t[r][3];
    }
    // corner_to_center
    double x = (cn[0][0] + cn[3][0] + cn[5][0] + cn[6][0]) / 4;
    double y = (cn[0][1] + cn[3][1] + cn[5][1] + cn[6][1]) / 4;
    double z = (cn[0][2] + cn[3][2] + cn[5][2] + cn[6][2]) / 4;
    double h = std::fabs(((cn[4][2] - cn[0][2]) + (cn[5][2] - cn[1][2]) + (cn[6][2] - cn[2][2]) + (cn[7][2] - cn[3][2])) / 4);
    auto d2 = [&](int a, int b) { return std::sqrt((cn[a][0] - cn[b][0]) * (cn[a][0] - cn[b][0]) + (cn[a][1] - cn[b][1]) * (cn[a][1] - cn[b][1])); };
    double l = (d2(0, 3) + d2(2, 1) + d2(4, 7) + d2(5, 6)) / 4;
    double w = (d2(0, 1) + d2(2, 3) + d2(4, 5) + d2(6, 7)) / 4;
    auto at = [&](int a, int b) { return std::atan2(cn[a][1] - cn[b][1], cn[a][0] - cn[b][0]); };
    double th = (at(1, 2) + at(0, 3) + at(5, 6) + at(4, 7)) / 4;
    // boxes_to_corners_3d: (dx, dy, dz) = (l, w, h) for both orders after the hwl reordering
    (void)hwl;
    const double dx = l, dy = w, dz = h;
    const float ca = static_cast<float>(std::cos(th)), sa = static_cast<float>(std::sin(th));
    for (int k = 0; k < 8; ++k) {
      const float px = static_cast<float>(sgn[k][0] / 2 * dx), py = static_cast<float>(sgn[k][1] / 2 * dy),
                  pz = static_cast<float>(sgn[k][2] / 2 * dz);
      const float rx = px * ca + py * (-sa), ry = px * sa + py * ca;  // points @ [[c, s], [-s, c]]
      o[(i * 8 + k) * 3 + 0] = static_cast<float>(static_cast<double>(rx) + x);
      o[(i * 8 + k) * 3 + 1] = static_cast<float>(static_cast<double>(ry) + y);
      o[(i * 8 + k) * 3 + 2] = static_cast<float>(static_cast<double>(pz) + z);
    }
  }
  return out;
}

// Late fusion and scoring of one ego sample. NMS compares the float32 IoU with float32(nms_thr), as NumPy does
// for an array against a Python float; the TP test compares in float64, as for a NumPy scalar.
// corners (N, 8, 3) and scores (N) are the concatenated sources;
// gt (M, 8, 3). Returns (scores sorted descending (n,), tp (n_thr, n) int8).
// OpenCOOD box filters, rotated NMS and GT-range mask. Returns indices into the input in NMS pick order and the
// BEV boxes of the input (indexed like the input).
std::vector<int> fuse_indices(const float* C, const float* S, py::ssize_t n0, float nms_thr, std::vector<Box2>& all) {
  // OpenCOOD filters
  std::vector<int> idx;
  idx.reserve(n0);
  for (py::ssize_t i = 0; i < n0; ++i) {
    float mn[3], mx[3];
    for (int r = 0; r < 3; ++r) { mn[r] = std::numeric_limits<float>::infinity(); mx[r] = -mn[r]; }
    for (int k = 0; k < 8; ++k)
      for (int r = 0; r < 3; ++r) {
        const float v = C[(i * 8 + k) * 3 + r];
        mn[r] = std::min(mn[r], v); mx[r] = std::max(mx[r], v);
      }
    const bool large_ok = (mx[0] - mn[0]) <= kMaxExtent && (mx[1] - mn[1]) <= kMaxExtent && (mx[2] - mn[2]) != 0.f;
    const bool z_ok = mn[2] >= kZMin && mx[2] <= kZMax;
    if (large_ok && z_ok) idx.push_back(static_cast<int>(i));
  }
  // rotated NMS: descending score (ascending stable sort reversed, as scores.argsort()[::-1]), top 1000
  std::vector<int> ord(idx.size());
  std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return S[idx[a]] < S[idx[b]]; });
  std::reverse(ord.begin(), ord.end());
  if (ord.size() > 1000) ord.resize(1000);
  std::vector<Box2> bx(idx.size());
  for (size_t k = 0; k < idx.size(); ++k) bx[k] = make_box(C + idx[k] * 24);
  all.assign(n0, Box2());
  for (size_t k = 0; k < idx.size(); ++k) all[idx[k]] = bx[k];
  std::vector<int> pick;
  std::vector<char> alive(ord.size(), 1);
  for (size_t a = 0; a < ord.size(); ++a) {
    if (!alive[a]) continue;
    pick.push_back(ord[a]);
    for (size_t b = a + 1; b < ord.size(); ++b)
      if (alive[b] && iou(bx[ord[a]], bx[ord[b]]) > nms_thr) alive[b] = 0;
  }
  // GT-range mask on the kept boxes
  std::vector<int> det;
  for (int k : pick) {
    const float* c = C + idx[k] * 24;
    bool in = true;
    for (int q = 0; q < 8 && in; ++q)
      in = c[3 * q] >= -kRangeX && c[3 * q] <= kRangeX && c[3 * q + 1] >= -kRangeY && c[3 * q + 1] <= kRangeY;
    if (in) det.push_back(idx[k]);
  }
  return det;
}

py::tuple fuse_and_score(f32arr corners, f32arr scores, f32arr gt, float nms_thr, f64arr thresholds) {
  const float* C = corners.data();
  const float* S = scores.data();
  std::vector<Box2> all;
  const std::vector<int> det = fuse_indices(C, S, corners.shape(0), nms_thr, all);
  // TP/FP: detections by descending score (stable), greedy best-IoU match among unmatched GT
  const py::ssize_t m = gt.shape(0);
  const float* Gc = gt.data();
  const auto T = thresholds.unchecked<1>();
  const int nt = static_cast<int>(T.shape(0));
  std::vector<int> o2(det.size());
  std::iota(o2.begin(), o2.end(), 0);
  std::stable_sort(o2.begin(), o2.end(), [&](int a, int b) { return S[det[a]] > S[det[b]]; });
  const py::ssize_t nd = static_cast<py::ssize_t>(det.size());
  f32arr sc_out(nd);
  py::array_t<int8_t> tp_out({py::ssize_t(nt), nd});
  float* so = sc_out.mutable_data();
  int8_t* to = tp_out.mutable_data();
  std::fill(to, to + nt * nd, 0);
  for (py::ssize_t k = 0; k < nd; ++k) so[k] = S[det[o2[k]]];
  if (m > 0 && nd > 0) {
    std::vector<Box2> gb(m);
    for (py::ssize_t j = 0; j < m; ++j) gb[j] = make_box(Gc + j * 24);
    std::vector<float> iou_m(nd * m);
    for (py::ssize_t k = 0; k < nd; ++k)
      for (py::ssize_t j = 0; j < m; ++j) iou_m[k * m + j] = iou(all[det[o2[k]]], gb[j]);
    std::vector<char> gt_alive(m);
    for (int t = 0; t < nt; ++t) {
      std::fill(gt_alive.begin(), gt_alive.end(), 1);
      py::ssize_t n_alive = m;
      for (py::ssize_t k = 0; k < nd && n_alive > 0; ++k) {
        // np.argmax over np.where(alive, iou, -1): first maximum, and NaN wins (first NaN)
        py::ssize_t best = -1;
        float bv = -std::numeric_limits<float>::infinity();
        for (py::ssize_t j = 0; j < m; ++j) {
          const float v = gt_alive[j] ? iou_m[k * m + j] : -1.f;
          if (std::isnan(v)) { best = j; bv = v; break; }
          if (best < 0 || v > bv) { best = j; bv = v; }
        }
        // float32 scalar vs Python float compares in float64 in NumPy; NaN never compares below
        if (!std::isnan(bv) && static_cast<double>(bv) < T(t)) continue;
        to[t * nd + k] = 1;
        gt_alive[best] = 0;
        --n_alive;
      }
    }
  }
  return py::make_tuple(sc_out, tp_out);
}

// Fused boxes of one sample (as fuse_boxes in eval_advercpm_fusion.py): corners (k, 8, 3) and scores in NMS order.
py::tuple fuse_boxes(f32arr corners, f32arr scores, float nms_thr) {
  const float* C = corners.data();
  const float* S = scores.data();
  std::vector<Box2> all;
  const std::vector<int> det = fuse_indices(C, S, corners.shape(0), nms_thr, all);
  const py::ssize_t k = static_cast<py::ssize_t>(det.size());
  f32arr oc({k, py::ssize_t(8), py::ssize_t(3)});
  f32arr os(k);
  for (py::ssize_t i = 0; i < k; ++i) {
    std::copy(C + det[i] * 24, C + det[i] * 24 + 24, oc.mutable_data() + i * 24);
    os.mutable_data()[i] = S[det[i]];
  }
  return py::make_tuple(oc, os);
}

// BEV IoU matrix (float32), as fast_eval.iou_matrix.
f32arr iou_matrix(f32arr a, f32arr b) {
  const py::ssize_t na = a.shape(0), nb = b.shape(0);
  f32arr out({na, nb});
  std::vector<Box2> ba(na), bb(nb);
  for (py::ssize_t i = 0; i < na; ++i) ba[i] = make_box(a.data() + i * 24);
  for (py::ssize_t j = 0; j < nb; ++j) bb[j] = make_box(b.data() + j * 24);
  float* o = out.mutable_data();
  for (py::ssize_t i = 0; i < na; ++i)
    for (py::ssize_t j = 0; j < nb; ++j) o[i * nb + j] = iou(ba[i], bb[j]);
  return out;
}

void set_filters(float range_x, float range_y, float z_min, float z_max, float max_extent) {
  kRangeX = range_x; kRangeY = range_y; kZMin = z_min; kZMax = z_max; kMaxExtent = max_extent;
}

py::tuple get_filters() { return py::make_tuple(kRangeX, kRangeY, kZMin, kZMax, kMaxExtent); }

PYBIND11_MODULE(fusion_core, m) {
  m.doc() = "Late fusion, rotated NMS and TP/FP matching for eval_advercpm_fusion.py";
  m.def("project", &project, py::arg("boxes"), py::arg("world_to_ego"), py::arg("hwl"));
  m.def("fuse_and_score", &fuse_and_score, py::arg("corners"), py::arg("scores"), py::arg("gt"),
        py::arg("nms_thr"), py::arg("thresholds"));
  m.def("fuse_boxes", &fuse_boxes, py::arg("corners"), py::arg("scores"), py::arg("nms_thr"));
  m.def("iou_matrix", &iou_matrix, py::arg("a"), py::arg("b"));
  m.def("set_filters", &set_filters, py::arg("range_x"), py::arg("range_y"), py::arg("z_min"), py::arg("z_max"),
        py::arg("max_extent"));
  m.def("get_filters", &get_filters);
}

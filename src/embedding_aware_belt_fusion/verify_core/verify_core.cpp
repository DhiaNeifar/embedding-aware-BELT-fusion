// AlignFormer's Verify stage and agreement rule in C++, for fast sweeps over cached Propose outputs.
//
// A port, not a variant: the default call reproduces, pair for pair,
//   alignformer_pva.refine.icp_refine   (mode icp_ransac, search off, optionally weighted)
//   alignformer_pva.refine.agree        (tolerance + consensus floor, ungated)
// as alignformer_pva.inference.estimate_correction chains them. The Abstain stage's Wald decision on the Propose
// estimate is NOT recomputed here: the caller passes `answered` (statistic > threshold), computed with the original
// abstain.abstention_threshold, so that rule has exactly one implementation.
//
// Everything is float32, mirroring the torch code's dtype and operation order where it matters (ties go to the first
// index, as torch.argmin does). RANSAC minimal samples for more than 16 hard pairs come from torch's own generator
// in the Python reference; the caller passes that table (it depends only on the pair count), so the draws are the
// same ones.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;
using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using i64arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kKabschEps = 1e-6f;    // procrustes.weighted_se2_kabsch eps
constexpr float kMinMatchMass = 1.0f;  // procrustes.MIN_MATCH_MASS
constexpr int kExhaustivePairs = 16;   // refine.EXHAUSTIVE_PAIRS

struct Box {
  float x, y, yaw, var;  // var: per-detection centre variance (variance_model.detection_variances), for weighting
};

struct Pose {
  float psi, tx, ty;
};

struct Config {
  std::vector<float> gates;
  int min_pairs;
  float inlier_m;
  bool weighted;
  float lam;            // heading_lambda
  float weight_ref;     // 2 * sigma_translation_m^2 (variance.precisions' reference)
  float agree_tol;      // AgreementConfig.tolerance_m
  int consensus_floor;  // AgreementConfig.consensus_floor
  float heading_weight;  // multiplier on the heading-tip points' Kabsch weight; 1 = the deployed fit
  bool gated;            // AgreementConfig.gated: Verify's fit only replaces a proposal the Wald test accepted
};

inline void moved(const Box& b, const Pose& p, float& mx, float& my) {
  const float c = std::cos(p.psi), s = std::sin(p.psi);
  mx = (c * b.x - s * b.y) + p.tx;
  my = (s * b.x + c * b.y) + p.ty;
}

// refine.mutual_nearest under pose p: ego index order, one-to-one.
void mutual_nearest(const std::vector<Box>& ego, const std::vector<Box>& cav, const Pose& p, float gate,
                    std::vector<int>& ego_idx, std::vector<int>& cav_idx) {
  ego_idx.clear();
  cav_idx.clear();
  const int n = (int)ego.size(), m = (int)cav.size();
  if (n == 0 || m == 0) return;
  std::vector<float> mx(m), my(m), d((size_t)n * m);
  for (int j = 0; j < m; ++j) moved(cav[j], p, mx[j], my[j]);
  // torch.cdist (default compute mode) switches to the matrix-product form ||x||^2 - 2 x.y + ||y||^2 when either
  // side has more than 25 rows; its rounding differs from the direct form, which decides near-ties.
  const bool mm = n > 25 || m > 25;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      if (mm) {
        const float xn = ego[i].x * ego[i].x + ego[i].y * ego[i].y, yn = mx[j] * mx[j] + my[j] * my[j];
        const float sq = (((-2.f * ego[i].x) * mx[j] + (-2.f * ego[i].y) * my[j]) + xn) + yn;
        d[(size_t)i * m + j] = std::sqrt(std::max(sq, 1e-30f));
      } else {
        const float dx = ego[i].x - mx[j], dy = ego[i].y - my[j];
        d[(size_t)i * m + j] = std::sqrt(dx * dx + dy * dy);
      }
    }
  std::vector<int> nearest_cav(n), nearest_ego(m);
  for (int i = 0; i < n; ++i) {
    int best = 0;
    for (int j = 1; j < m; ++j)
      if (d[(size_t)i * m + j] < d[(size_t)i * m + best]) best = j;
    nearest_cav[i] = best;
  }
  for (int j = 0; j < m; ++j) {
    int best = 0;
    for (int i = 1; i < n; ++i)
      if (d[(size_t)i * m + j] < d[(size_t)best * m + j]) best = i;
    nearest_ego[j] = best;
  }
  for (int i = 0; i < n; ++i) {
    const int j = nearest_cav[i];
    if (nearest_ego[j] == i && d[(size_t)i * m + j] < gate) {
      ego_idx.push_back(i);
      cav_idx.push_back(j);
    }
  }
}

// procrustes.weighted_se2_kabsch over (target p, source q, weight w).
Pose kabsch(const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& qx,
            const std::vector<float>& qy, const std::vector<float>& w) {
  const size_t n = w.size();
  float mass = 0.f;
  for (size_t k = 0; k < n; ++k) mass += std::max(w[k], 0.f);
  const float safe = std::max(mass, kKabschEps);
  float pbx = 0.f, pby = 0.f, qbx = 0.f, qby = 0.f;
  for (size_t k = 0; k < n; ++k) {
    const float wk = std::max(w[k], 0.f);
    pbx += wk * px[k];
    pby += wk * py[k];
    qbx += wk * qx[k];
    qby += wk * qy[k];
  }
  pbx /= safe; pby /= safe; qbx /= safe; qby /= safe;
  float cross = 0.f, dot = 0.f;
  for (size_t k = 0; k < n; ++k) {
    const float wk = std::max(w[k], 0.f);
    const float dpx = px[k] - pbx, dpy = py[k] - pby, dqx = qx[k] - qbx, dqy = qy[k] - qby;
    cross += wk * (dqx * dpy - dqy * dpx);
    dot += wk * (dqx * dpx + dqy * dpy);
  }
  const float magnitude = std::sqrt(cross * cross + dot * dot);
  const bool resolvable = magnitude > kKabschEps;
  if (!(mass >= kMinMatchMass && resolvable)) return {0.f, 0.f, 0.f};
  const float psi = std::atan2(cross, dot);
  const float c = std::cos(psi), s = std::sin(psi);
  return {psi, pbx - (c * qbx - s * qby), pby - (s * qbx + c * qby)};
}

// refine._exact_solve: heading-augmented (centres, then heading tips), CAV heading folded onto the ego partner's
// half-plane; weights from refine.pair_weights (scalar mode: the centre precision for both points) or ones.
Pose exact_solve(const std::vector<Box>& ego, const std::vector<Box>& cav, const std::vector<int>& ei,
                 const std::vector<int>& ci, const Config& cfg) {
  const size_t k = ei.size();
  std::vector<float> px(2 * k), py(2 * k), qx(2 * k), qy(2 * k), w(2 * k, 1.f);
  for (size_t a = 0; a < k; ++a) {
    const Box& e = ego[ei[a]];
    const Box& c = cav[ci[a]];
    float cyaw = c.yaw;
    if (std::cos(cyaw - e.yaw) < 0.f) cyaw = cyaw + kPi;
    px[a] = e.x; py[a] = e.y;
    qx[a] = c.x; qy[a] = c.y;
    px[k + a] = e.x + cfg.lam * std::cos(e.yaw);
    py[k + a] = e.y + cfg.lam * std::sin(e.yaw);
    qx[k + a] = c.x + cfg.lam * std::cos(cyaw);
    qy[k + a] = c.y + cfg.lam * std::sin(cyaw);
    if (cfg.weighted) {
      const float rho = cfg.weight_ref / (e.var + c.var);
      w[a] = rho;
      w[k + a] = rho;
    }
    if (cfg.heading_weight != 1.f) w[k + a] = w[k + a] * cfg.heading_weight;
  }
  return kabsch(px, py, qx, qy, w);
}

// refine.robust_exact_solve: two-pair RANSAC over the hard pairs, then the exact fit over the consensus.
Pose robust_exact_solve(const std::vector<Box>& ego, const std::vector<Box>& cav, const std::vector<int>& ei,
                        const std::vector<int>& ci, const Config& cfg, const std::vector<std::vector<int>>& table,
                        int& consensus, std::vector<int>& kept_e, std::vector<int>& kept_c) {
  const int count = (int)ei.size();
  if (count < 3) {
    consensus = count;
    kept_e = ei;
    kept_c = ci;
    return exact_solve(ego, cav, ei, ci, cfg);
  }
  std::vector<int> sa, sb;
  if (count <= kExhaustivePairs) {
    for (int i = 0; i < count; ++i)
      for (int j = i + 1; j < count; ++j) { sa.push_back(i); sb.push_back(j); }
  } else {
    if (count >= (int)table.size() || table[count].empty())
      throw std::runtime_error("no RANSAC sample table for " + std::to_string(count) + " hard pairs");
    const std::vector<int>& t = table[count];
    for (size_t s = 0; s + 1 < t.size(); s += 2) { sa.push_back(t[s]); sb.push_back(t[s + 1]); }
  }
  std::vector<float> tx(count), ty(count), sx(count), sy(count);
  for (int a = 0; a < count; ++a) {
    tx[a] = ego[ei[a]].x; ty[a] = ego[ei[a]].y;
    sx[a] = cav[ci[a]].x; sy[a] = cav[ci[a]].y;
  }
  int best_count = -1;
  float best_total = std::numeric_limits<float>::infinity();
  std::vector<char> best_inliers(count, 0), inl(count);
  for (size_t s = 0; s < sa.size(); ++s) {
    const int i = sa[s], j = sb[s];
    const float ctx = tx[j] - tx[i], cty = ty[j] - ty[i], csx = sx[j] - sx[i], csy = sy[j] - sy[i];
    const float psi = std::atan2(cty, ctx) - std::atan2(csy, csx);
    const float c = std::cos(psi), sn = std::sin(psi);
    const float msx = (sx[i] + sx[j]) / 2.f, msy = (sy[i] + sy[j]) / 2.f;
    const float mtx = (tx[i] + tx[j]) / 2.f, mty = (ty[i] + ty[j]) / 2.f;
    const float ttx = mtx - (c * msx - sn * msy), tty = mty - (sn * msx + c * msy);
    int n = 0;
    float total = 0.f;
    for (int a = 0; a < count; ++a) {
      const float dx = ((c * sx[a] - sn * sy[a]) + ttx) - tx[a];
      const float dy = ((sn * sx[a] + c * sy[a]) + tty) - ty[a];
      const float r = std::sqrt(dx * dx + dy * dy);
      inl[a] = r < cfg.inlier_m;
      if (inl[a]) { ++n; total += r; }
    }
    // Most inliers; ties to the smallest inlier residual sum; then the first sample.
    if (n > best_count || (n == best_count && total < best_total)) {
      best_count = n;
      best_total = total;
      best_inliers = inl;
    }
  }
  std::vector<int> kei, kci;
  for (int a = 0; a < count; ++a)
    if (best_inliers[a]) { kei.push_back(ei[a]); kci.push_back(ci[a]); }
  if ((int)kei.size() < 2) { kei = ei; kci = ci; }
  consensus = (int)kei.size();
  kept_e = kei;
  kept_c = kci;
  return exact_solve(ego, cav, kei, kci, cfg);
}

struct Refined {
  Pose pose;
  bool engaged;
  int consensus;
  std::vector<int> inlier_e, inlier_c;  // the hard pairs behind the final fit
};

// refine._refine_one (search off): ICP with a shrinking gate, each stage a RANSAC re-solve from the noisy boxes.
Refined refine_one(Pose p, const std::vector<Box>& ego, const std::vector<Box>& cav, const Config& cfg,
                   const std::vector<std::vector<int>>& table) {
  Refined out{p, false, 0, {}, {}};
  std::vector<int> ei, ci;
  for (float gate : cfg.gates) {
    mutual_nearest(ego, cav, out.pose, gate, ei, ci);
    if ((int)ei.size() < cfg.min_pairs) break;
    out.pose = robust_exact_solve(ego, cav, ei, ci, cfg, table, out.consensus, out.inlier_e, out.inlier_c);
    out.engaged = true;
  }
  return out;
}

inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

std::vector<Box> read_boxes(const float* rows, int64_t begin, int64_t end) {
  std::vector<Box> out;
  out.reserve(end - begin);
  for (int64_t r = begin; r < end; ++r) out.push_back({rows[r * 4], rows[r * 4 + 1], rows[r * 4 + 2], rows[r * 4 + 3]});
  return out;
}

// Verify + agreement over P pairs. Boxes are (rows, 4) = x, y, yaw, centre variance, pair p owning rows
// [offsets[p], offsets[p+1]). `propose` is (P, 3) = psi, tx, ty; `answered` is the Wald decision on it.
// Returns (decided (P, 3), refined (P, 3), engaged (P,), consensus (P,), inliers (Q, 2) = (ego, cav) box index of the
// final fit's hard pairs, inlier_off (P+1,)).
py::tuple verify_pairs(f32arr ego, i64arr ego_off, f32arr cav, i64arr cav_off, f32arr propose,
                       py::array_t<bool, py::array::c_style | py::array::forcecast> answered,
                       std::vector<float> gates, int min_pairs, float inlier_m, bool weighted, float lam,
                       float weight_ref, float agree_tol, int consensus_floor,
                       std::vector<std::vector<int>> ransac_table, float heading_weight, bool gated) {
  const int64_t P = propose.shape(0);
  if (ego_off.shape(0) != P + 1 || cav_off.shape(0) != P + 1 || answered.shape(0) != P)
    throw std::runtime_error("offsets / answered do not match the number of pairs");
  Config cfg{gates, min_pairs, inlier_m, weighted, lam, weight_ref, agree_tol, consensus_floor, heading_weight, gated};
  const float* er = ego.data();
  const float* cr = cav.data();
  const int64_t* eo = ego_off.data();
  const int64_t* co = cav_off.data();
  const float* pr = propose.data();
  const bool* ans = answered.data();

  py::array_t<float> decided({P, (int64_t)3}), refined({P, (int64_t)3});
  py::array_t<bool> engaged(P);
  py::array_t<int64_t> consensus(P);
  float* dr = decided.mutable_data();
  float* rr = refined.mutable_data();
  bool* en = engaged.mutable_data();
  int64_t* cs = consensus.mutable_data();
  std::vector<int64_t> inl_off(P + 1, 0);
  std::vector<int> inl_flat;
  {
    py::gil_scoped_release release;
    for (int64_t p = 0; p < P; ++p) {
      const std::vector<Box> e = read_boxes(er, eo[p], eo[p + 1]);
      const std::vector<Box> c = read_boxes(cr, co[p], co[p + 1]);
      const Pose soft{pr[p * 3], pr[p * 3 + 1], pr[p * 3 + 2]};
      const Refined r = refine_one(soft, e, c, cfg, ransac_table);
      // abstain.decide(estimate) in "abstain" mode: the proposal or nothing.
      const Pose fallback = ans[p] ? soft : Pose{0.f, 0.f, 0.f};
      Pose out = fallback;
      if (r.engaged) {
        const float dx = r.pose.tx - soft.tx, dy = r.pose.ty - soft.ty;
        const float gap = std::sqrt(dx * dx + dy * dy) + cfg.lam * std::fabs(wrap(r.pose.psi - soft.psi));
        bool agreeing = gap <= cfg.agree_tol;
        if (cfg.consensus_floor > 0) agreeing = agreeing || r.consensus >= cfg.consensus_floor;
        if (cfg.gated) agreeing = agreeing && ans[p];
        out = agreeing ? r.pose : Pose{0.f, 0.f, 0.f};
      }
      dr[p * 3] = out.psi; dr[p * 3 + 1] = out.tx; dr[p * 3 + 2] = out.ty;
      rr[p * 3] = r.pose.psi; rr[p * 3 + 1] = r.pose.tx; rr[p * 3 + 2] = r.pose.ty;
      en[p] = r.engaged;
      cs[p] = r.consensus;
      for (size_t q = 0; q < r.inlier_e.size(); ++q) {
        inl_flat.push_back(r.inlier_e[q]);
        inl_flat.push_back(r.inlier_c[q]);
      }
      inl_off[p + 1] = inl_off[p] + (int64_t)r.inlier_e.size();
    }
  }
  py::array_t<int64_t> inliers({(int64_t)(inl_flat.size() / 2), (int64_t)2});
  int64_t* ip = inliers.mutable_data();
  for (size_t q = 0; q < inl_flat.size(); ++q) ip[q] = inl_flat[q];
  py::array_t<int64_t> inlier_off(P + 1);
  std::copy(inl_off.begin(), inl_off.end(), inlier_off.mutable_data());
  return py::make_tuple(decided, refined, engaged, consensus, inliers, inlier_off);
}

}  // namespace

PYBIND11_MODULE(verify_core, m) {
  m.doc() = "AlignFormer Verify (icp_ransac) + agreement rule over cached Propose outputs";
  m.def("verify_pairs", &verify_pairs, py::arg("ego"), py::arg("ego_off"), py::arg("cav"), py::arg("cav_off"),
        py::arg("propose"), py::arg("answered"), py::arg("gates"), py::arg("min_pairs"), py::arg("inlier_m"),
        py::arg("weighted"), py::arg("lam"), py::arg("weight_ref"), py::arg("agree_tol"),
        py::arg("consensus_floor"), py::arg("ransac_table"), py::arg("heading_weight") = 1.0f,
        py::arg("gated") = false);
}

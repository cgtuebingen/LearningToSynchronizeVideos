// Patrick Wieschollek, 2016
#include <malloc.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

template <class T>
inline constexpr T pow8(const T base) {
  return (((base * base) * (base * base)) * ((base * base) * (base * base)));
}

/*
python setup.py clean && python setup.py build && python setup.py install --user
*/

template <typename T>
class Mat {
 public:
  T* data;
  int width, height;

  Mat(T* d, const int h, const int w) : data(d), width(w), height(h) {}
  Mat() : width(0), height(0) {}
  Mat(const int h, const int w) : width(w), height(h) { data = new T[w * h]; }

  inline T& operator()(int x, int y) { return data[x * width + y]; }

  inline const T& operator()(int x, int y) const { return data[x * width + y]; }

  const int col_arg_min(int y) {
    T min_value = data[0 * width + y];
    int min_x = 0;

    for (int i = 1; i < height; ++i) {
      float cur_value = data[i * width + y];
      if (cur_value < min_value) {
        min_value = cur_value;
        min_x = i;
      }
    }
    return min_x;
  }
};

class Dijkstra {
  Mat<float> cost;
  int m, n;

 public:
  float* latest_prop;

  Dijkstra(float* cost_, int m_, int n_) : m(m_), n(n_) {
    // std::cout << "[Dijkstra]   load cost of size " << m_<< " x "<< n_<<
    // std::endl;
    cost = Mat<float>(cost_, m_, n_);
    // std::cout << "alive01" << std::endl;
  }

  /**
   * @brief extract boxes with side at most 400 around tour
   * @details note, each box has a patch from top-left to bottom-right
   *
   * @param tour coarse tour
   * @param max_frames max side of box
   *
   * @return box-corners (top, bottom, left, right)
   */
  std::vector<std::array<int, 4> > find_boxes(
      std::vector<std::array<int, 2> >& tour, int max_frames = 400) {
    std::vector<std::array<int, 4> > boxes;
    int r_last, r_start, s_last, s_start;
    r_last = r_start = tour[0][0];
    s_last = s_start = tour[0][1];

    for (unsigned int i = 0, i_e = tour.size(); i < i_e; ++i) {
      const int r = tour[i][0];
      const int s = tour[i][1];

      if ((r > r_start + max_frames) || (s > s_start + max_frames)) {
        if ((r > r_start + 2 * max_frames) || (s > s_start + 2 * max_frames)) {
          boxes.push_back({r_start, r_last, s_start, s_last});
        } else {
          boxes.push_back({r_start, r, s_start, s});
        }
        r_start = r;
        s_start = s;
      }
      r_last = r;
      s_last = s;
    }

    if ((r_last < r_start + 2 * max_frames) &&
        (s_last < s_start + 2 * max_frames)) {
      boxes.push_back({r_start, r_last, s_start, s_last});
    }
    return boxes;
  }
  /**
   * @brief runs dijkstra for each stripe individually without any assumptions
   * @details stripes are overlapping, reject stripes which are not connected
   * onall ends cost should be at least 88% less than nearby paths
   *
   * @param stripe_len length of each stripe
   * @param frame_offset tolerance when test connectivity
   *
   * @return all found tour points sorted(!)
   */
  std::vector<std::array<int, 2> > multiple_tours(int stripe_len = 250,
                                                  int frame_offset = 30) {
    // 83.33 = 250*10 / 30 --> 2*60*30/10 = 360.00  (2min)
    std::vector<std::array<int, 2> > single_tour;
    std::vector<std::vector<std::array<int, 2> > > multi_tour;

    const int overlap = stripe_len / 2;
    // overlap = 0;
    // std::cout << "alive02" << std::endl;
    for (int iter = 0; iter < n; iter += stripe_len) {
      const int start = (iter - overlap < 0) ? 0 : (iter - overlap);
      const int end =
          (iter + stripe_len + overlap > n) ? n : (iter + stripe_len + overlap);
      multi_tour.push_back(
          tour(start, end, overlap, iter / stripe_len, n / stripe_len));
    }
    // std::cout << "alive03" << std::endl;

    for (unsigned int i = 0, i_e = n / stripe_len; i < i_e; ++i) {
      // test connection between stripes
      // std::cout << "alive03-" << i << " " << i_e << std::endl;

      // std::cout << "alive03- " << multi_tour[i].size() << std::endl;

      if (i == 0) {
        if (std::abs(multi_tour[i].front()[0] - multi_tour[i + 1].back()[0]) <
            frame_offset)
          single_tour.insert(single_tour.end(), multi_tour[i].rbegin(),
                             multi_tour[i].rend());
      }
      // std::cout << "alive03-b" << std::endl;
      if (i > 0 && i < i_e - 1) {
        if ((std::abs(multi_tour[i].front()[0] - multi_tour[i + 1].back()[0]) <
                 frame_offset &&
             std::abs(multi_tour[i - 1].front()[0] - multi_tour[i].back()[0]) <
                 frame_offset))
          single_tour.insert(single_tour.end(), multi_tour[i].rbegin(),
                             multi_tour[i].rend());
      }
      // std::cout << "alive03-c" << std::endl;
      if ((i == i_e - 1) && (i > 0)) {
        if (std::abs(multi_tour[i - 1].front()[0] - multi_tour[i].back()[0]) <
            frame_offset)
          single_tour.insert(single_tour.end(), multi_tour[i].rbegin(),
                             multi_tour[i].rend());
      }
      // std::cout << "alive 05" << std::endl;
    }

    // std::cout << "alive04 " << single_tour.size() << std::endl;
    std::vector<std::array<int, 2> > result;
    for (unsigned int i = 0, i_e = single_tour.size(); i < i_e; ++i) {
      const float c1 = pow8(cost(single_tour[i][0], single_tour[i][1]));
      float c2 = 0;
      int x_off = 50;
      int y_off = 50;

      if (single_tour[i][0] > m / 2) x_off = -50;
      if (single_tour[i][1] > n / 2) y_off = -50;
      c2 = pow8(cost(single_tour[i][0] + x_off, single_tour[i][1] + y_off));

      if (c1 / c2 < 0.89) result.push_back(single_tour[i]);
    }
    // std::cout << "alive05" << std::endl;
    if ((unsigned int)n / 4 > result.size()) result.clear();

    return result;
  }

  /**
   * @brief Run Dijkstra's Algorithm on a single stripe
   * @details [long description]
   *
   * @param off offset from left
   * @param col_end last column in matrix
   * @param overlap_info the overlap value (just as an information)
   * @param id_info stripe id in loop (just as an information)
   * @param max_info number of all stripes (just as an information)
   * @param ssp known start points ? (for fine resolution we know the start and
   * endpoints)
   * @return [description]
   */
  std::vector<std::array<int, 2> > tour(
      const int off = 0, const int col_end = 0, const int overlap_info = 0,
      const int id_info = 0, const int max_info = 0, const bool ssp = false) {
    int cut_start = 0;
    int cut_end = col_end;

    if (id_info == 0) {
      cut_end = col_end - overlap_info;
    } else if (id_info == max_info) {
      cut_start = off + overlap_info;
    } else {
      cut_start = off + overlap_info;
      cut_end = col_end - overlap_info;
    }

    // std::cout << "[Dijkstra]   compute tour from "<< off << " to " << col_end
    // << std::endl;

    Mat<int> directions(m, (col_end - off));
    Mat<float> accumulate_costs(m, (col_end - off));

    for (int i = 0; i < m * (col_end - off); ++i) {
      directions.data[i] = 0;
    }

    // information about the start
    for (int r = 0; r < m; ++r) accumulate_costs(r, 0) = pow8(cost(r, off));
    if (ssp) {
      // we have to start on the top left corner
      // accumulate_costs(0, 0) = 0;
    }

    // init first row
    for (int c = 1; c < (col_end - off); ++c) {
      accumulate_costs(0, c) =
          accumulate_costs(0, c - 1) + pow8(cost(0, off + c));
    }

    for (int r = 1; r < m; ++r) {
      for (int c = 1; c < (col_end - off); ++c) {
        const float cur_cost = pow8(cost(r, off + c));

        // add top-left-costs
        accumulate_costs(r, c) = accumulate_costs(r - 1, c - 1) + cur_cost;
        directions(r, c) = 2;

        // test top
        float tmp = accumulate_costs(r - 1, c) + cur_cost;
        if (tmp < accumulate_costs(r, c)) {
          accumulate_costs(r, c) = tmp;
          directions(r, c) = 1;
        }
        // test left
        tmp = accumulate_costs(r, c - 1) + cur_cost;
        if (tmp < accumulate_costs(r, c)) {
          accumulate_costs(r, c) = tmp;
          directions(r, c) = 3;
        }
      }
    }

    latest_prop = accumulate_costs.data;

    // // find tour
    std::vector<std::array<int, 2> > t;
    t.clear();

    int column = col_end;  // column
    int row = accumulate_costs.col_arg_min(col_end - off - 1);
    if (ssp) {
      // we have to end on the bottom right corner
      row = m - 1;
    }

    // std::cout << "[Dijkstra]   start in " << row<< " / "<< column
    // <<std::endl;

    if (column > cut_start)
      if (column < cut_end) t.push_back({row, column});

    while ((row > 0) && (column > off + 1)) {
      switch (directions(row, column - off - 1)) {
        case 1:  // from top
          row--;
          break;
        case 2:  // from top-left
          column--;
          row--;
          break;
        case 3:  // from left
          column--;
          break;
      }

      if ((column > cut_start) && (column < cut_end))
        t.push_back({row, column});
    }
    return t;
  }
};

void advanced_dijkstra(int width, int frame_accurarcy, float* cost, int m,
                       int n, int** tour, int* tour_len, int** box,
                       int* box_len) {
  Dijkstra alg(cost, m, n);
  std::vector<std::array<int, 2> > tour_vec =
      alg.multiple_tours(width, frame_accurarcy);

  // std::cout << "found "<< tour_vec.size() <<" tour entries" << std::endl;
  int tour_len_ = tour_vec.size() * 2;
  int* tour_;

  if (tour_len_ == 0) {
    tour_ = (int*)malloc(1 * sizeof(int));
  } else {
    tour_ = (int*)malloc(tour_len_ * sizeof(int));
    for (unsigned int i = 0, i_e = tour_vec.size(); i < i_e; i++) {
      tour_[2 * i] = tour_vec[i][0];
      tour_[2 * i + 1] = tour_vec[i][1];
    }
  }

  *tour = tour_;
  *tour_len = tour_len_;

  ////////////////////////////////////////
  // std::cout << "get boxes" << std::endl;
  int box_len_ = 0;
  int* box_;
  if (tour_len_ == 0) {
    box_len_ = 1;
    box_ = (int*)malloc(box_len_ * sizeof(int));

  } else {
    std::vector<std::array<int, 4> > box_vec = alg.find_boxes(tour_vec);

    box_len_ = box_vec.size() * 4;
    box_ = (int*)malloc(box_len_ * sizeof(int));

    for (unsigned int i = 0, i_e = box_vec.size(); i < i_e; i++) {
      box_[4 * i] = box_vec[i][0];
      box_[4 * i + 1] = box_vec[i][1];
      box_[4 * i + 2] = box_vec[i][2];
      box_[4 * i + 3] = box_vec[i][3];
    }
  }

  *box = box_;
  *box_len = box_len_;
}

void plain_dijkstra(int width, int frame_accurarcy, float* cost, int m, int n,
                    int** tour, int* tour_len) {
  Dijkstra alg(cost, m, n);
  std::vector<std::array<int, 2> > tour_vec = alg.tour(0, n, 0, 0, 0, true);

  int tour_len_ = tour_vec.size() * 2;
  int* tour_ = (int*)malloc(tour_len_ * sizeof(int));

  for (unsigned int i = 0, i_e = tour_vec.size(); i < i_e; i++) {
    tour_[2 * i] = tour_vec[i][0];
    tour_[2 * i + 1] = tour_vec[i][1];
  }

  *tour = tour_;
  *tour_len = tour_len_;
}

void accumulate(float* cost, int m, int n) {
  Dijkstra alg(cost, m, n);
  std::vector<std::array<int, 2> > tour_vec = alg.tour(0, n, 0, 0, 0, true);

  for (unsigned int i = 0; i < m * n; ++i) {
    cost[i] = alg.latest_prop[i];
  }
}
